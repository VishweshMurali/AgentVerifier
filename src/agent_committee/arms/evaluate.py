"""Score arms on the DEV eval split only and write arms/dev_report.json.

    python -m agent_committee.arms.evaluate --arms tfidf [--out PATH]

Refuses to overwrite an existing report. LLM arms are built through a lazy factory
`agent_committee.arms.<prompt|agent>.make_arm(pool, book)` (the module owns its LLMClient).
"""

import argparse
import json
import sys
from pathlib import Path

from agent_committee.arms.base import Arm
from agent_committee.arms.labels import build_label_book
from agent_committee.arms.splits import ARMS_SPLIT_SEED, make_arms_split
from agent_committee.arms.tfidf import TfidfArm
from agent_committee.data import load_dev
from agent_committee.schemas import Case

REPORT_PATH = Path(__file__).with_name("dev_report.json")
SIMPLER = ("tfidf", "prompt")  # arms the agent has to beat


def score(arm: Arm, cases: list[Case]) -> dict:
    preds = [arm.predict(c.id, c.text) for c in cases]
    correct = sum(p.pred == c.label for p, c in zip(preds, cases, strict=True))
    return {
        "n": len(cases),
        "correct": correct,
        "accuracy": correct / len(cases),
        "unparseable": sum(p.pred is None for p in preds),
        "prompt_tokens": sum(p.prompt_tokens for p in preds),
        "completion_tokens": sum(p.completion_tokens for p in preds),
    }


def best_simpler(results: dict[str, dict]) -> str | None:
    """Highest-accuracy simpler arm among those evaluated; ties go to the earlier in SIMPLER."""
    have = [a for a in SIMPLER if a in results]
    return max(have, key=lambda a: (results[a]["accuracy"], -SIMPLER.index(a))) if have else None


def build_report(results: dict[str, dict], counts: dict[str, int]) -> dict:
    return {
        "split_seed": ARMS_SPLIT_SEED,
        "split_counts": counts,
        "scored_on": "eval",
        "results": results,
        "best_simpler_arm": best_simpler(results),
    }


def write_report(path: Path, report: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build(name: str, pool: list[Case]) -> Arm:
    if name == "tfidf":
        arm = TfidfArm()
        arm.fit(pool)
        return arm
    if name in ("prompt", "agent"):
        book = build_label_book(pool)
        if name == "prompt":
            from agent_committee.arms.prompt import make_arm
        else:
            from agent_committee.arms.agent import make_arm
        return make_arm(pool, book)
    raise ValueError(f"unknown arm {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="tfidf")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"{args.out} exists; refusing to overwrite", file=sys.stderr)
        return 1
    split = make_arms_split(load_dev("data"))
    results = {}
    for name in args.arms.split(","):
        results[name] = score(_build(name, split.pool), split.eval)
        print(name, results[name])
    write_report(args.out, build_report(results, split.counts()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
