"""Throughput benchmark: how many LLM calls/hour does the local Ollama model sustain?

Uses DEV texts only. Independent of the Phase B wrapper on purpose (no cache: we want real latency).
The figure is for one short call per case, not a full arm (shortlist, few-shot examples, up to 4
calls per case), so it is an upper bound on arm throughput.
If Ollama is unreachable or answers with an error, this raises OllamaUnreachable; a response
without token usage raises BenchmarkFailed. The CLI exits non-zero; it never fakes.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, NotFoundError, OpenAI

from agent_committee.llm import BASE_URL, MODEL
from agent_committee.schemas import Case

TIMEOUT_S = 120.0


class BenchmarkFailed(RuntimeError):
    """The benchmark could not produce a trustworthy figure."""


class OllamaUnreachable(BenchmarkFailed):
    """Ollama cannot be reached, the model is missing, or the server answered with an error."""


@dataclass(frozen=True)
class BenchmarkResult:
    n_calls: int
    elapsed_s: float
    concurrency: int
    calls_per_hour: float
    mean_latency_s: float
    prompt_tokens: int
    completion_tokens: int


def calls_per_hour(n_calls: int, elapsed_s: float) -> float:
    if n_calls < 0 or elapsed_s <= 0:
        raise ValueError("need n_calls >= 0 and elapsed_s > 0")
    return n_calls / elapsed_s * 3600.0


def _tokens(usage: Any, field: str) -> int:
    value = getattr(usage, field, None) if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise BenchmarkFailed(f"response has no valid token usage '{field}': {usage!r}")
    if value < 0:
        raise BenchmarkFailed(f"response token usage '{field}' is negative: {value!r}")
    return int(value)


def run_benchmark(
    cases: list[Case], *, concurrency: int = 1, client: Any = None
) -> BenchmarkResult:
    """`client` is for unit tests (a fake with `chat.completions.create`); runtime uses Ollama."""
    if not cases:
        raise ValueError("no cases to benchmark: pass at least one DEV case")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    client = client or OpenAI(base_url=BASE_URL, api_key="ollama", timeout=TIMEOUT_S, max_retries=0)

    def one(case: Case) -> tuple[float, int, int]:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=16,
            messages=[
                {"role": "system", "content": "Classify the banking intent in a few words."},
                {"role": "user", "content": case.text},
            ],
        )
        latency = time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        return latency, _tokens(usage, "prompt_tokens"), _tokens(usage, "completion_tokens")

    try:
        one(cases[0])  # warm-up: loads the model into memory, excluded from timing
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(one, cases))
        elapsed = time.perf_counter() - start
    except APIConnectionError as exc:  # includes APITimeoutError
        raise OllamaUnreachable(f"cannot reach Ollama at {BASE_URL}: {exc}") from exc
    except NotFoundError as exc:
        raise OllamaUnreachable(
            f"model '{MODEL}' or endpoint not found at {BASE_URL} (is the model pulled?): {exc}"
        ) from exc
    except APIStatusError as exc:
        raise OllamaUnreachable(
            f"Ollama at {BASE_URL} answered HTTP {exc.status_code}: {exc}"
        ) from exc

    return BenchmarkResult(
        n_calls=len(rows),
        elapsed_s=elapsed,
        concurrency=concurrency,
        calls_per_hour=calls_per_hour(len(rows), elapsed),
        mean_latency_s=sum(r[0] for r in rows) / len(rows),
        prompt_tokens=sum(r[1] for r in rows),
        completion_tokens=sum(r[2] for r in rows),
    )
