"""LLM wrapper for the candidate arms: SQLite cache, retries, timeout, token + latency logging.

Runtime always talks to Ollama (OpenAI-compatible endpoint). If it cannot be reached after the
retries, `LLMUnavailable` is raised; there is no fallback and no mock. Unit tests inject a fake
`transport`; production code never does.
"""

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:7b"
TIMEOUT_S = 60.0
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0

Transport = Callable[[dict[str, Any]], dict[str, Any]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    latency_s REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    cached INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    wall_s REAL NOT NULL,
    latency_s REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    ts REAL NOT NULL
);
"""


class LLMUnavailable(RuntimeError):
    """Ollama unreachable or failing after all retries. Callers must stop, never fake."""


@dataclass(frozen=True)
class LLMResult:
    message: dict[str, Any]  # the assistant message dict: content, tool_calls, ...
    prompt_tokens: int
    completion_tokens: int
    latency_s: float  # latency of the real call (for cache hits: the original call's latency)
    cached: bool
    attempts: int

    @property
    def content(self) -> str:
        return self.message.get("content") or ""


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


class LLMClient:
    def __init__(
        self,
        db_path: str | Path = "cache/llm.sqlite",
        *,
        model: str = MODEL,
        transport: Transport | None = None,
        max_retries: int = MAX_RETRIES,
        backoff_base_s: float = BACKOFF_BASE_S,
        timeout_s: float = TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._transport = transport or ollama_transport(timeout=timeout_s)
        self._max_retries = max_retries
        self._backoff = backoff_base_s
        self._sleep = sleep
        self._lock = threading.Lock()
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
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
        key = request_key(request)

        t_start = time.perf_counter()
        with self._lock:
            row = self._db.execute(
                "SELECT response_json, prompt_tokens, completion_tokens, latency_s "
                "FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row:
            result = LLMResult(json.loads(row[0]), row[1], row[2], row[3], True, 0)
            self._log(key, result, time.perf_counter() - t_start)
            return result

        response, attempts, latency = self._call_with_retries(request)
        usage = response.get("usage") or {}
        message = response["choices"][0]["message"]
        result = LLMResult(
            message,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            latency,
            False,
            attempts,
        )
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    json.dumps(request, sort_keys=True),
                    json.dumps(message),
                    result.prompt_tokens,
                    result.completion_tokens,
                    latency,
                    time.time(),
                ),
            )
            self._db.commit()
        self._log(key, result, time.perf_counter() - t_start)
        return result

    def _call_with_retries(self, request: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
        last: Exception | None = None
        for attempt in range(1, self._max_retries + 2):
            t0 = time.perf_counter()
            try:
                response = self._transport(request)
                return response, attempt, time.perf_counter() - t0
            except Exception as exc:  # classified below; non-retryable errors are re-raised
                if not _is_retryable(exc):
                    raise
                last = exc
                if attempt <= self._max_retries:
                    self._sleep(self._backoff * 2 ** (attempt - 1))
        raise LLMUnavailable(
            f"LLM call failed after {self._max_retries + 1} attempts: {last!r}"
        ) from last

    def _log(self, key: str, r: LLMResult, wall_s: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO calls (key, cached, attempts, wall_s, latency_s, prompt_tokens, "
                "completion_tokens, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    int(r.cached),
                    r.attempts,
                    wall_s,
                    r.latency_s,
                    r.prompt_tokens,
                    r.completion_tokens,
                    time.time(),
                ),
            )
            self._db.commit()

    def usage_summary(self) -> dict[str, float]:
        """Totals over all logged calls (cache hits carry their original tokens/latency)."""
        with self._lock:
            n, hits, pt, ct, lat = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(cached), 0), COALESCE(SUM(prompt_tokens), 0), "
                "COALESCE(SUM(completion_tokens), 0), COALESCE(AVG(latency_s), 0) FROM calls"
            ).fetchone()
        return {
            "calls": n,
            "cache_hits": hits,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "mean_latency_s": lat,
        }
