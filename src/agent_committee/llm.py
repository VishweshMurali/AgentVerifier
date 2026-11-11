"""LLM wrapper for the candidate arms: SQLite cache, retries, timeout, token + latency logging.

Runtime always talks to Ollama (OpenAI-compatible endpoint). If it cannot be reached after the
retries, `LLMUnavailable` is raised; there is no fallback and no mock. Unit tests inject a fake
`transport`; production code never does.

Every call is attributed to a `phase` ("build" = arm construction/DEV evaluation, "run" = the
Phase D test run) and an `arm`. Both are part of the cache key, so build-phase and run-phase
prompts never share rows and the leakage check can scan `phase = 'build'` rows on their own.

Cached answers are replayed verbatim on reruns, which hides any Ollama non-determinism by design.
The key has the model tag; pass `model_digest` to also pin the exact weights.
"""

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    NotFoundError,
    OpenAI,
)

DEFAULT_BASE_URL = "http://localhost:11434/v1"


# the one place the Ollama address is set; a container points it at the host with
# OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
def _base_url(value: str | None) -> str:
    """The OpenAI-compatible endpoint; accepts the address with or without /v1 (Open WebUI and
    others set OLLAMA_BASE_URL without it)."""
    url = (value or DEFAULT_BASE_URL).rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


BASE_URL = _base_url(os.environ.get("OLLAMA_BASE_URL"))
MODEL = "llama3.1:8b"
TIMEOUT_S = 60.0
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0
BUSY_TIMEOUT_S = 30.0

Phase = Literal["build", "run"]
Transport = Callable[[dict[str, Any]], dict[str, Any]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    phase TEXT NOT NULL,
    arm TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    finish_reason TEXT,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    latency_s REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    phase TEXT NOT NULL,
    arm TEXT NOT NULL,
    status TEXT NOT NULL,
    cached INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    wall_s REAL NOT NULL,
    latency_s REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    finish_reason TEXT,
    error TEXT,
    ts REAL NOT NULL
);
"""


class LLMUnavailable(RuntimeError):
    """Ollama unreachable, model missing, or failing after all retries. Callers must stop."""


_REQUIRED_COLUMNS = {
    "cache": {"key", "phase", "arm", "finish_reason", "prompt_tokens", "completion_tokens"},
    "calls": {"key", "phase", "arm", "status", "finish_reason", "error"},
}


class LLMCacheSchemaError(RuntimeError):
    """The SQLite cache file was created by an older, incompatible schema."""


class LLMUsageMissing(RuntimeError):
    """The response carried no token usage. Cost accounting must not silently become zero."""


class MalformedResponse(RuntimeError):
    """The model answered, but not in the OpenAI chat shape (no choices, content not a string or
    None, tool_calls not a list of objects). Raised for live and for cached replies alike.

    `prompt_tokens` / `completion_tokens` are what the response reported (None if it reported
    none); the same numbers are written to the failure row of the `calls` ledger.
    """

    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class LLMCacheWriteError(RuntimeError):
    """The model answered but the cache/ledger write failed (after the busy timeout). The paid
    answer is not stored. `prompt_tokens` / `completion_tokens` carry what was spent."""

    def __init__(self, message: str, *, prompt_tokens: int, completion_tokens: int) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class ModelDigestMismatch(RuntimeError):
    """Ollama serves different weights than the configured `model_digest`. The cache is neither
    read nor written. `configured` and `served` hold the two (normalised) digests."""

    def __init__(self, message: str, *, configured: str, served: str) -> None:
        super().__init__(message)
        self.configured = configured
        self.served = served


@dataclass(frozen=True)
class LLMResult:
    message: dict[str, Any]  # the assistant message dict: content, tool_calls, ...
    prompt_tokens: int
    completion_tokens: int
    latency_s: float  # latency of the real call (for cache hits: the original call's latency)
    cached: bool
    attempts: int
    finish_reason: str | None = None

    @property
    def content(self) -> str:
        return self.message.get("content") or ""

    @property
    def truncated(self) -> bool:
        """True when generation stopped at max_tokens; the content is incomplete."""
        return self.finish_reason == "length"


def request_key(request: dict[str, Any]) -> str:
    """SHA-256 over the canonical full request (model, messages, tools, temperature, ...)."""
    canon = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError | APITimeoutError):
        return True
    return isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500)


def ollama_transport(base_url: str = BASE_URL, timeout: float = TIMEOUT_S) -> Transport:
    client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout, max_retries=0)

    def call(request: dict[str, Any]) -> dict[str, Any]:
        return client.chat.completions.create(**request).model_dump()

    return call


def _normalise_digest(digest: str) -> str:
    return digest.strip().lower().removeprefix("sha256:")


def fetch_served_digest(
    model: str, base_url: str = BASE_URL, *, http_client: httpx.Client | None = None
) -> str:
    """Digest of `model` as Ollama serves it now (GET /api/tags). Any failure is LLMUnavailable."""
    url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        if http_client is not None:
            resp = http_client.get(url)
        else:
            with httpx.Client(timeout=10.0) as http:
                resp = http.get(url)
        resp.raise_for_status()
        for entry in resp.json()["models"]:
            if entry.get("name") == model or entry.get("model") == model:
                digest = entry["digest"]
                if isinstance(digest, str) and digest:
                    return digest
                raise ValueError(f"empty digest for {model!r}")
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise LLMUnavailable(f"cannot resolve the served digest of '{model}': {exc!r}") from exc
    raise LLMUnavailable(f"model '{model}' is not listed by Ollama at {url} (is it pulled?)")


def validate_message(
    message: Any, *, prompt_tokens: int | None, completion_tokens: int | None
) -> dict[str, Any]:
    """Boundary check of an assistant message; raises MalformedResponse (usage attached)."""

    def bad(why: str) -> MalformedResponse:
        return MalformedResponse(
            f"malformed model response: {why}",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    if not isinstance(message, dict):
        raise bad(f"message is not an object: {message!r}")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise bad(f"content is neither a string nor null: {content!r}")
    calls = message.get("tool_calls")
    if calls is not None and (
        not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls)
    ):
        raise bad(f"tool_calls is not a list of objects: {calls!r}")
    return message


def _token_count(usage: dict[str, Any], field: str) -> int:
    value = usage.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise LLMUsageMissing(f"response usage has no valid '{field}': {usage!r}")
    if value < 0:
        raise LLMUsageMissing(f"response usage '{field}' is negative: {value!r}")
    return int(value)


class LLMClient:
    def __init__(
        self,
        db_path: str | Path = "cache/llm.sqlite",
        *,
        phase: Phase,
        model: str = MODEL,
        model_digest: str | None = None,
        transport: Transport | None = None,
        max_retries: int = MAX_RETRIES,
        backoff_base_s: float = BACKOFF_BASE_S,
        timeout_s: float = TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
        busy_timeout_s: float = BUSY_TIMEOUT_S,
        served_digest: Callable[[str], str] | None = None,
    ) -> None:
        if phase not in ("build", "run"):
            raise ValueError("phase must be 'build' or 'run'")
        self.phase: Phase = phase
        self.model = model
        self.model_digest = model_digest
        # F5: a pinned digest must match what Ollama serves, before the cache is touched. A
        # production client (no injected transport) resolves it from Ollama; a test that injects a
        # transport may pass `served_digest` (a fake tags endpoint) or skip the check.
        resolver = served_digest or (fetch_served_digest if transport is None else None)
        if model_digest is not None and resolver is not None:
            served = resolver(model)
            if _normalise_digest(served) != _normalise_digest(model_digest):
                raise ModelDigestMismatch(
                    f"Ollama serves '{model}' with digest {served}, not the configured "
                    f"{model_digest}: refusing to read or write the cache",
                    configured=_normalise_digest(model_digest),
                    served=_normalise_digest(served),
                )
        self._transport = transport or ollama_transport(timeout=timeout_s)
        self._max_retries = max_retries
        self._backoff = backoff_base_s
        self._sleep = sleep
        self._lock = threading.Lock()
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: no implicit transactions; writes use an explicit BEGIN IMMEDIATE
        # (see _tx), so a connection never upgrades a read to a write (that fails at once).
        self._db = sqlite3.connect(
            str(db_path), check_same_thread=False, timeout=busy_timeout_s, isolation_level=None
        )
        self._db.execute(f"PRAGMA busy_timeout={int(busy_timeout_s * 1000)}")
        if str(db_path) != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")  # readers no longer block the writer
            self._db.execute("PRAGMA synchronous=NORMAL")
        self._check_schema()
        self._db.executescript(_SCHEMA)

    def _check_schema(self) -> None:
        """Refuse a cache file written by an older schema; its rows have no phase/arm tags."""
        for table, required in _REQUIRED_COLUMNS.items():
            cols = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
            if cols and not required <= cols:
                self._db.close()
                raise LLMCacheSchemaError(
                    f"cache table '{table}' is missing columns {sorted(required - cols)}: the "
                    "file predates the phase/arm schema. Move or delete it and re-run."
                )

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        arm: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = 0,
    ) -> LLMResult:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            request["tools"] = tools
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if seed is not None:
            request["seed"] = seed
        keyed = {**request, "_phase": self.phase, "_arm": arm, "_digest": self.model_digest}
        key = request_key(keyed)

        t_start = time.perf_counter()
        row = self._lookup(key)
        if row:
            try:
                validate_message(json.loads(row[0]), prompt_tokens=row[1], completion_tokens=row[2])
            except (MalformedResponse, ValueError) as exc:
                bad = exc
                if not isinstance(exc, MalformedResponse):
                    bad = MalformedResponse(
                        f"malformed cached reply: {exc}",
                        prompt_tokens=row[1],
                        completion_tokens=row[2],
                    )
                self._log_failure_safe(
                    key, arm, 0, time.perf_counter() - t_start, bad, row[1], row[2]
                )
                raise bad from None
            result = self._to_result(row, cached=True, attempts=0)
            self._log(key, arm, "ok", result, time.perf_counter() - t_start)
            return result

        state = {"attempts": 0}
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        try:
            response, latency = self._call_with_retries(request, state)
            usage = response.get("usage") or {}
            if not isinstance(usage, dict):  # e.g. a list or number: no usage to account (R6-L1)
                raise LLMUsageMissing(f"response usage is not an object: {usage!r}")
            prompt_tokens = _token_count(usage, "prompt_tokens")
            completion_tokens = _token_count(usage, "completion_tokens")
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise MalformedResponse(
                    f"malformed model response: no usable choices: {choices!r}",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            choice = choices[0]
            message = validate_message(
                choice.get("message"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            message_json = json.dumps(message)
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and not isinstance(finish_reason, str):
                # checked before anything is stored: SQLite cannot bind it (R6-L1)
                raise MalformedResponse(
                    f"malformed model response: finish_reason {finish_reason!r}",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
        except Exception as exc:
            self._log_failure_safe(
                key,
                arm,
                state["attempts"],
                time.perf_counter() - t_start,
                exc,
                prompt_tokens,
                completion_tokens,
            )
            raise

        try:
            # One transaction: cache row, stored-row re-read and calls row land together or not at
            # all. INSERT OR IGNORE then re-read: concurrent callers on one key all return the same
            # stored answer instead of the last writer winning.
            with self._tx():
                self._db.execute(
                    "INSERT OR IGNORE INTO cache (key, phase, arm, request_json, response_json, "
                    "finish_reason, prompt_tokens, completion_tokens, latency_s, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        key,
                        self.phase,
                        arm,
                        json.dumps(request, sort_keys=True),
                        message_json,
                        finish_reason,
                        prompt_tokens,
                        completion_tokens,
                        latency,
                        time.time(),
                    ),
                )
                stored = self._db.execute(
                    "SELECT response_json, finish_reason FROM cache WHERE key = ?", (key,)
                ).fetchone()
                assert stored is not None
                # The answer is the stored one (so concurrent callers agree); the usage and latency
                # logged are what THIS request measured, not the winner's.
                result = LLMResult(
                    json.loads(stored[0]),
                    prompt_tokens,
                    completion_tokens,
                    latency,
                    False,
                    state["attempts"],
                    stored[1],
                )
                self._insert_call_row(
                    self._call_values(key, arm, "ok", result, time.perf_counter() - t_start)
                )
        except sqlite3.Error as exc:
            self._log_failure_safe(
                key,
                arm,
                state["attempts"],
                time.perf_counter() - t_start,
                exc,
                prompt_tokens,
                completion_tokens,
            )
            raise LLMCacheWriteError(
                f"model answered but the cache/ledger write failed: {exc!r}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ) from exc
        return result

    @contextmanager
    def _tx(self) -> Iterator[None]:
        """One write transaction (BEGIN IMMEDIATE ... COMMIT), rolled back on any failure so the
        connection never stays inside an open transaction."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.execute("COMMIT")
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    def _lookup(self, key: str) -> tuple | None:
        with self._lock:
            return self._db.execute(
                "SELECT response_json, prompt_tokens, completion_tokens, latency_s, finish_reason "
                "FROM cache WHERE key = ?",
                (key,),
            ).fetchone()

    @staticmethod
    def _to_result(row: tuple, *, cached: bool, attempts: int) -> LLMResult:
        return LLMResult(json.loads(row[0]), row[1], row[2], row[3], cached, attempts, row[4])

    def _call_with_retries(
        self, request: dict[str, Any], state: dict[str, int]
    ) -> tuple[dict[str, Any], float]:
        last: Exception | None = None
        for attempt in range(1, self._max_retries + 2):
            state["attempts"] = attempt
            t0 = time.perf_counter()
            try:
                return self._transport(request), time.perf_counter() - t0
            except NotFoundError as exc:
                raise LLMUnavailable(f"model or endpoint not found: {exc}") from exc
            except Exception as exc:  # classified below; non-retryable errors are re-raised
                if not _is_retryable(exc):
                    raise
                last = exc
                if attempt <= self._max_retries:
                    self._sleep(self._backoff * 2 ** (attempt - 1))
        raise LLMUnavailable(
            f"LLM call failed after {self._max_retries + 1} attempts: {last!r}"
        ) from last

    def _call_values(self, key: str, arm: str, status: str, r: LLMResult, wall_s: float) -> tuple:
        return (
            key,
            self.phase,
            arm,
            status,
            int(r.cached),
            r.attempts,
            wall_s,
            r.latency_s,
            r.prompt_tokens,
            r.completion_tokens,
            r.finish_reason,
            None,
        )

    def _log(self, key: str, arm: str, status: str, r: LLMResult, wall_s: float) -> None:
        with self._tx():
            self._insert_call_row(self._call_values(key, arm, status, r, wall_s))

    def _log_failure_safe(
        self,
        key: str,
        arm: str,
        attempts: int,
        wall_s: float,
        exc: BaseException,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        """Write the failure row; if that write fails too, keep the ORIGINAL error and attach the
        logging failure to it as a note."""
        try:
            with self._tx():
                self._insert_call_row(
                    (
                        key,
                        self.phase,
                        arm,
                        "failed",
                        0,
                        attempts,
                        wall_s,
                        0.0,
                        prompt_tokens or 0,
                        completion_tokens or 0,
                        None,
                        repr(exc),
                    )
                )
        except sqlite3.Error as log_exc:
            exc.add_note(f"failure log write also failed: {log_exc!r}")

    def _insert_call_row(self, values: tuple) -> None:
        self._db.execute(
            "INSERT INTO calls (key, phase, arm, status, cached, attempts, wall_s, latency_s, "
            "prompt_tokens, completion_tokens, finish_reason, error, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*values, time.time()),
        )

    def usage_summary(self) -> dict[str, float]:
        """Totals over all logged calls; cache hits carry their original tokens/latency."""
        with self._lock:
            n, ok, failed, hits, pt, ct, lat = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(status = 'ok'), 0), "
                "COALESCE(SUM(status = 'failed'), 0), COALESCE(SUM(cached), 0), "
                "COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), "
                "COALESCE(AVG(CASE WHEN status = 'ok' THEN latency_s END), 0) FROM calls"
            ).fetchone()
        return {
            "calls": n,
            "ok": ok,
            "failed": failed,
            "cache_hits": hits,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "mean_latency_s": lat,
        }
