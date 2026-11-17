"""A hosted agent model for quick tests (owner approved 2026-11-14): Groq, for the public site.

Set AGENT_PROVIDER=groq and GROQ_API_KEY (optionally GROQ_MODEL). The LLM arms then run on Groq's
OpenAI-compatible endpoint instead of local Ollama. Honesty rules:
- a hosted model is NOT pinned: the provider may change its weights, so a re-run may differ.
  plan.json, the audit and the page say "hosted, not pinned"; no digest is checked;
- Ollama (llama3.1:8b, digest-pinned) stays the reference agent when AGENT_PROVIDER is unset;
- the sealed run never uses this module (only quick.py imports it; it is outside arms/, so the
  frozen arms and the saved-model keys are unchanged).

The free tier allows few tokens per minute, so calls are paced under the limit and a 429 waits
for the time the provider names. A wait longer than MAX_WAIT_S (the daily limit) stops the run:
a starved agent must not turn into a verdict.

Things that differ from the Ollama reference, all recorded in plan.json (`settings()`):
- the agent's first model call may be forced to use a tool (`tool_choice`, GROQ_FIRST_TOOL=
  required|named). It is OFF: measured on 2026-11-17 it made this model worse (see
  docs/HOSTED_AGENT_TOOL_NOTE.md). The setting is part of the cache key (`FIRST_TOOL_KEY`);
- the model reasons before it answers and its reasoning counts against the arms' small reply
  limits, so it is asked for the least reasoning (`reasoning_effort`, GROQ_REASONING);
- Groq refuses a reply whose tool call is not valid JSON (HTTP 400 `tool_use_failed`) instead of
  returning it. That is the model's own malformed tool call, so it is handed to the agent as one
  and scores wrong, like any other (locked agent rule). Groq reports no token counts for it, so
  the request's estimated size and the full reply limit are charged (never zero).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agent_committee.llm import TIMEOUT_S, LLMClient, LLMUnavailable, Phase, Transport

PROVIDER_ENV = "AGENT_PROVIDER"
GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TPM = 8000  # free tier, tokens per minute (override: GROQ_TPM)
HEADROOM = 0.92  # stay under the limit: the estimate of a request's size is rough
CHARS_PER_TOKEN = 3.5
REPLY_TOKENS = 300  # assumed reply size when the request sets no max_tokens
MAX_WAIT_S = 90.0  # a longer wait is the daily limit, not the per-minute one
RATE_RETRIES = 6
NOT_PINNED = "hosted agent (not pinned)"
REASONING = "low"
TOOL_USE_FAILED = "tool_use_failed"
PROXY_URL_ENV = "GROQ_PROXY_URL"  # optional key-holding proxy (proxy/worker.js), ends in /v1
PROXY_TOKEN_ENV = "GROQ_PROXY_TOKEN"  # the shared secret the proxy checks (never logged or saved)
PROXY_HEADER = "x-proxy-token"
FIRST_TOOL_ENV = "GROQ_FIRST_TOOL"  # required | named to force the first tool call (off by default)
# a label that only varies the cache key: answers cached before the first call was forced (or from
# a run with it off) are never replayed as if they came from this setting
FIRST_TOOL_KEY = "hosted:first-call-tool"
SEARCH_FIRST_ENV = "GROQ_SEARCH_FIRST"  # on by default; 0 = do not add the search-first line
CACHE_TAG_ENV = "GROQ_CACHE_TAG"  # measuring only: a label that keeps repeat runs from replaying
SEARCH_FIRST_TEXT = (
    "\nImportant: always call search_examples first, even when you already think you know the "
    "label. Never reply with a label before you have called a tool."
)


def provider() -> str | None:
    """'groq' when a hosted agent is configured, None for the local Ollama reference."""
    value = os.environ.get(PROVIDER_ENV, "").strip().lower()
    if value in ("", "ollama"):
        return None
    if value != "groq":
        raise ValueError(f"{PROVIDER_ENV} must be 'groq' or 'ollama', not {value!r}")
    return value


def model() -> str:
    return os.environ.get("GROQ_MODEL", "").strip() or GROQ_MODEL


def label() -> str:
    """How the agent is named on the page, in plan.json and in the audit."""
    return f"{model()} via Groq (hosted, not pinned)"


def reasoning() -> str | None:
    """The reasoning effort asked for: GROQ_REASONING when set ('none' sends no setting), else
    the least for the gpt-oss models. Other models are sent no setting (they may refuse it)."""
    named = os.environ.get("GROQ_REASONING", "").strip().lower()
    if named:
        return None if named == "none" else named
    return REASONING if "gpt-oss" in model() else None


def first_call_choice(env: Mapping[str, str] | None = None) -> dict[str, Any] | str | None:
    """The `tool_choice` sent on the agent's first model call: None (free, the default),
    "required" (GROQ_FIRST_TOOL=required) or the named search tool (GROQ_FIRST_TOOL=named).

    Forcing is OFF because it was measured to hurt (2026-11-17, docs/HOSTED_AGENT_TOOL_NOTE.md):
    on a case it is sure of, the hosted model writes the label as plain text instead of a tool
    call, Groq rejects that as `tool_use_failed`, and the case scores malformed. Left free, the
    same reply is sent back once (the locked agent rule) and the model then uses its tools."""
    env = os.environ if env is None else env
    value = (env.get(FIRST_TOOL_ENV) or "").strip().lower()
    if value in ("1", "required", "yes", "on", "true"):
        return "required"
    if value == "named":
        return {"type": "function", "function": {"name": "search_examples"}}
    return None


def first_call_tool(env: Mapping[str, str] | None = None) -> bool:
    return first_call_choice(env) is not None


def search_first(env: Mapping[str, str] | None = None) -> bool:
    """True (the default, owner approved 2026-11-17) when the hosted agent's first call carries the
    extra search-first instruction. Measured on the same 39-message sample: without it the tool-use
    floor failed twice (82.6%, 4 sent back of 23); with it, twice, 100% and no send-backs, same
    accuracy (docs/HOSTED_AGENT_TOOL_NOTE.md). GROQ_SEARCH_FIRST=0 turns it off."""
    env = os.environ if env is None else env
    return (env.get(SEARCH_FIRST_ENV) or "1").strip().lower() not in ("0", "off", "no", "false")


def with_search_first(request: dict[str, Any]) -> dict[str, Any]:
    """The request with the extra instruction added to its system message (a copy)."""
    messages = [dict(m) for m in request["messages"]]
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = m["content"] + SEARCH_FIRST_TEXT
            break
    return {**request, "messages": messages}


def first_call_key() -> str | None:
    """The label that varies the cache key with the hosted settings that change the model's input
    or what it must do (None when none is set, so the plain keys stay as they were)."""
    parts = []
    choice = first_call_choice()
    if choice is not None:
        parts.append(f"{FIRST_TOOL_KEY}:{'named' if isinstance(choice, dict) else choice}")
    if search_first():
        parts.append("search-first")
    if tag := (os.environ.get(CACHE_TAG_ENV) or "").strip():
        parts.append(f"tag-{tag}")
    return "|".join(parts) or None


def is_first_call(request: dict[str, Any]) -> bool:
    """A request that offers tools before the model has said or used anything yet."""
    if not request.get("tools"):
        return False
    return not any(m.get("role") in ("assistant", "tool") for m in request.get("messages") or [])


def proxy(env: Mapping[str, str] | None = None) -> tuple[str, str] | None:
    """(base url, token) of the key-holding proxy when one is configured, else None (the engine
    then talks to Groq itself with GROQ_API_KEY). Both settings or neither; https only; the URL
    ends in /v1 because the client adds /chat/completions."""
    env = os.environ if env is None else env
    url = (env.get(PROXY_URL_ENV) or "").strip().rstrip("/")
    token = (env.get(PROXY_TOKEN_ENV) or "").strip()
    if not url and not token:
        return None
    if not url or not token:
        raise ValueError(f"{PROXY_URL_ENV} and {PROXY_TOKEN_ENV} must be set together")
    if not url.startswith("https://") or not url.endswith("/v1"):
        raise ValueError(f"{PROXY_URL_ENV} must be an https address ending in /v1")
    return url, token


def connection() -> tuple[str, str, dict[str, str]]:
    """(base url, api key, extra headers) for the client. With a proxy the engine never holds the
    Groq key: the SDK gets a placeholder and the proxy's token travels in its own header."""
    via = proxy()
    if via is not None:
        return via[0], "via-proxy", {PROXY_HEADER: via[1]}
    return GROQ_URL, api_key(), {}


def settings() -> dict[str, Any]:
    """What a run on the hosted agent records in plan.json."""
    out: dict[str, Any] = {
        "kind": "hosted",
        "model": label(),
        "provider": "groq",
        "pinned": False,
        "reasoning_effort": reasoning(),
        "refused_tool_call": "scored as a malformed tool call; tokens estimated",
        "search_first": (
            "the first call adds: always call search_examples first, never reply with a label first"
            if search_first()
            else "off"
        ),
        "first_call": (
            f"forced for experiment: tool_choice {first_call_choice()}"
            if first_call_tool()
            else "free: the model may answer without a tool (it is then sent back once)"
        ),
    }
    if proxy() is not None:
        out["via_proxy"] = True  # only that a proxy is used, never its address or token
    return out


def refused_tool_call(exc: Exception, request: dict[str, Any]) -> dict[str, Any] | None:
    """Groq's 400 for a tool call the model wrote badly, as the reply it stands for: one tool
    call whose arguments are the model's raw text (the agent finds it malformed). None for any
    other error."""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(error, dict) or error.get("code") != TOOL_USE_FAILED:
        return None
    raw = error.get("failed_generation")
    limit = int(request.get("max_tokens") or REPLY_TOKENS)
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "refused",
                            "type": "function",
                            "function": {
                                "name": TOOL_USE_FAILED,
                                "arguments": raw if isinstance(raw, str) else "",
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": max(0, estimate_tokens(request) - limit),
            "completion_tokens": limit,
        },
    }


def api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise LLMUnavailable("GROQ_API_KEY is not set; the hosted agent cannot run")
    return key


def estimate_tokens(request: dict[str, Any]) -> int:
    """A rough size of one request plus its reply, for pacing only (never for cost)."""
    sent = json.dumps([request.get("messages"), request.get("tools")], ensure_ascii=False)
    return int(len(sent) / CHARS_PER_TOKEN) + int(request.get("max_tokens") or REPLY_TOKENS)


class Pacer:
    """Keeps the tokens sent in any 60 seconds under the limit by waiting before a call."""

    def __init__(
        self,
        tokens_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.limit = max(1, int(tokens_per_minute * HEADROOM))
        self._clock = clock
        self._sleep = sleep
        self._spent: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _used(self, now: float) -> int:
        while self._spent and now - self._spent[0][0] >= 60.0:
            self._spent.popleft()
        return sum(n for _, n in self._spent)

    def wait(self, tokens: int) -> float:
        """Wait until `tokens` more fit in the window; returns the seconds waited."""
        waited = 0.0
        with self._lock:
            while True:
                now = self._clock()
                # a request larger than the whole window still goes, once the window is empty
                if not self._spent or self._used(now) + tokens <= self.limit:
                    return waited
                pause = max(0.05, 60.0 - (now - self._spent[0][0]))
                self._sleep(pause)
                waited += pause

    def spend(self, tokens: int) -> None:
        with self._lock:
            self._spent.append((self._clock(), max(0, int(tokens))))


def retry_after(exc: Exception) -> float | None:
    """Seconds the provider asks us to wait (a 429's retry-after header), if it names any."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _tpm() -> int:
    try:
        return max(1, int(os.environ.get("GROQ_TPM", GROQ_TPM)))
    except ValueError:
        return GROQ_TPM


def groq_transport(
    *,
    timeout: float = TIMEOUT_S,
    pacer: Pacer | None = None,
    create: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Transport:
    """One OpenAI-shaped call to Groq, paced under the per-minute limit. `create` is injected in
    tests (it stands in for client.chat.completions.create)."""
    from openai import (
        AuthenticationError,
        BadRequestError,
        OpenAI,
        PermissionDeniedError,
        RateLimitError,
    )

    if create is None:
        base_url, key, headers = connection()
        client = OpenAI(
            base_url=base_url,
            api_key=key,
            default_headers=headers or None,
            timeout=timeout,
            max_retries=0,
        )
        create = client.chat.completions.create
    pace = pacer or Pacer(_tpm())
    effort = reasoning()
    extra = {"reasoning_effort": effort} if effort else {}
    first_choice = first_call_choice()
    nudge = search_first()
    # the size guess is rough and errs high (it reserves a full reply); once real counts come back,
    # guess is scaled to them, so the pacer does not wait for room the request never used. A request
    # that proves bigger than guessed raises the scale again, and a 429 is still waited out.
    scale = {"v": 1.0}

    def call(request: dict[str, Any]) -> dict[str, Any]:
        raw_guess = estimate_tokens(request)
        guess = max(1, int(raw_guess * scale["v"]))
        for _ in range(RATE_RETRIES + 1):
            pace.wait(guess)
            try:
                first = is_first_call(request)
                forced = {"tool_choice": first_choice} if first_choice is not None and first else {}
                sent = with_search_first(request) if nudge and first else request
                out = create(**sent, **extra, **forced)
            except BadRequestError as exc:
                refused = refused_tool_call(exc, request)
                if refused is None:
                    raise
                pace.spend(guess)
                return refused
            except (AuthenticationError, PermissionDeniedError) as exc:
                raise LLMUnavailable(
                    "the hosted agent refused our credentials (check the Groq key, or the "
                    "proxy token): the test cannot run"
                ) from exc
            except RateLimitError as exc:
                wait = retry_after(exc)
                if wait is not None and wait > MAX_WAIT_S:
                    raise LLMUnavailable(
                        f"Groq's limit is used up for now (it asks to wait {wait:.0f} s): "
                        "the hosted agent cannot finish this test"
                    ) from exc
                pace.spend(guess)  # the refused request still counts against the window
                sleep(min(MAX_WAIT_S, wait if wait is not None else 10.0))
                continue
            data = out.model_dump() if hasattr(out, "model_dump") else out
            usage = data.get("usage") if isinstance(data, dict) else None
            total = usage.get("total_tokens") if isinstance(usage, dict) else None
            pace.spend(total if isinstance(total, int) else guess)
            if isinstance(total, int) and raw_guess > 0:
                seen = min(1.25, max(0.5, total / raw_guess))
                scale["v"] = min(1.25, max(0.5, 0.7 * scale["v"] + 0.3 * seen))
            return data
        raise LLMUnavailable("Groq kept refusing the request for its rate limit")

    return call


def client(db_path: str | Path = "cache/llm.sqlite", *, phase: Phase = "build") -> LLMClient:
    """The LLM client for the hosted agent: same cache and ledger, no digest pin."""
    return LLMClient(
        db_path,
        phase=phase,
        model=model(),
        model_digest=first_call_key(),
        transport=groq_transport(),
    )
