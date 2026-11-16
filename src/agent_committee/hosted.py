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

Two things differ from the Ollama reference, both recorded in plan.json (`settings()`):
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
from collections.abc import Callable
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


def settings() -> dict[str, Any]:
    """What a run on the hosted agent records in plan.json."""
    return {
        "kind": "hosted",
        "model": label(),
        "provider": "groq",
        "pinned": False,
        "reasoning_effort": reasoning(),
        "refused_tool_call": "scored as a malformed tool call; tokens estimated",
    }


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
    from openai import BadRequestError, OpenAI, RateLimitError

    if create is None:
        client = OpenAI(base_url=GROQ_URL, api_key=api_key(), timeout=timeout, max_retries=0)
        create = client.chat.completions.create
    pace = pacer or Pacer(_tpm())
    effort = reasoning()
    extra = {"reasoning_effort": effort} if effort else {}
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
                out = create(**request, **extra)
            except BadRequestError as exc:
                refused = refused_tool_call(exc, request)
                if refused is None:
                    raise
                pace.spend(guess)
                return refused
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
        db_path, phase=phase, model=model(), model_digest=None, transport=groq_transport()
    )
