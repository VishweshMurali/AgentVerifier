"""Throughput benchmark: how many LLM calls/hour does the local Ollama model sustain?

Uses DEV texts only. Independent of the Phase B wrapper on purpose (no cache: we want real latency).
If Ollama is unreachable this raises OllamaUnreachable and the CLI exits non-zero; it never fakes.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from openai import APIConnectionError, OpenAI

from agent_committee.schemas import Case

BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.1:8b"


class OllamaUnreachable(RuntimeError):
    pass


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


def run_benchmark(cases: list[Case], *, concurrency: int = 1) -> BenchmarkResult:
    client = OpenAI(base_url=BASE_URL, api_key="ollama", timeout=120, max_retries=0)

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
        usage = resp.usage
        return (
            time.perf_counter() - t0,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    try:
        one(cases[0])  # warm-up: loads the model into memory, excluded from timing
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(one, cases))
        elapsed = time.perf_counter() - start
    except APIConnectionError as exc:
        raise OllamaUnreachable(f"cannot reach Ollama at {BASE_URL}: {exc}") from exc

    return BenchmarkResult(
        n_calls=len(rows),
        elapsed_s=elapsed,
        concurrency=concurrency,
        calls_per_hour=calls_per_hour(len(rows), elapsed),
        mean_latency_s=sum(r[0] for r in rows) / len(rows),
        prompt_tokens=sum(r[1] for r in rows),
        completion_tokens=sum(r[2] for r in rows),
    )
