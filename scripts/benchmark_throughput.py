"""CLI: measure Ollama calls/hour on DEV texts. Exits 2 if Ollama is unreachable (never mocks)."""

import argparse
import sys

from agent_committee.data import load_dev
from agent_committee.throughput import OllamaUnreachable, run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=30, help="number of DEV cases to call")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    cases = load_dev()[: args.n]
    try:
        result = run_benchmark(cases, concurrency=args.concurrency)
    except OllamaUnreachable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"calls:            {result.n_calls} (concurrency {result.concurrency})")
    print(f"elapsed:          {result.elapsed_s:.1f}s")
    print(f"mean latency:     {result.mean_latency_s:.2f}s")
    print(f"tokens in/out:    {result.prompt_tokens}/{result.completion_tokens}")
    print(f"calls/hour:       {result.calls_per_hour:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
