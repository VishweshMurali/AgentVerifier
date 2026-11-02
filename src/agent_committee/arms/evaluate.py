"""Score arms on the DEV eval split only and write arms/dev_report.json.

    python -m agent_committee.arms.evaluate --arms linear,prompt,agent [--out PATH]

Refuses to overwrite an existing report. One linear layer is fitted once and shared by all arms.
LLM arms are built through a lazy factory
`agent_committee.arms.<prompt|agent>.make_arm(pool, book, layer)` (the module owns its LLMClient).
"""

import argparse
import json
import sys
from pathlib import Path

from agent_committee.arms import config
from agent_committee.arms.base import Arm
from agent_committee.arms.labels import build_label_book
from agent_committee.arms.linear import Decision, LinearArm, LinearLayer, make_layer
from agent_committee.arms.splits import ARMS_SPLIT_SEED, make_arms_split
from agent_committee.data import load_dev
from agent_committee.schemas import Case

REPORT_PATH = Path(__file__).with_name("dev_report.json")
SIMPLER = ("linear", "prompt")  # arms the agent has to beat


def score(arm: Arm, cases: list[Case], decisions: list[Decision]) -> dict:
    """One arm's results. `decisions` are the shared layer's, aligned with `cases`."""
    preds = [arm.predict(c.id, c.text) for c in cases]
    right = [p.pred == c.label for p, c in zip(preds, cases, strict=True)]
    flagged = [ok for ok, d in zip(right, decisions, strict=True) if d.null]
    n = len(cases)
    return {
        "n": n,
        "correct": sum(right),
        "accuracy": sum(right) / n,
        "unparseable": sum(p.pred is None for p in preds),
        "prompt_tokens": sum(p.prompt_tokens for p in preds),
        "completion_tokens": sum(p.completion_tokens for p in preds),
        "flagged_cases": len(flagged),
        "flagged_correct": sum(flagged),
        "flagged_accuracy": sum(flagged) / len(flagged) if flagged else None,
        "own_tool_rate": sum(p.own_tool_calls > 0 and not p.sent_back for p in preds) / n,
        "send_back_rate": sum(p.sent_back for p in preds) / n,
        "malformed_rate": sum(p.malformed for p in preds) / n,
    }


def layer_report(layer: LinearLayer, cases: list[Case], decisions: list[Decision]) -> dict:
    n = len(cases)
    covered = sum(c.label in d.shortlist for c, d in zip(cases, decisions, strict=True))
    return {
        "fingerprint": layer.fingerprint(),
        "null_threshold": layer.threshold,
        "shortlist_k": layer.shortlist_k,
        "flagged_rate": sum(d.null for d in decisions) / n,
        "shortlist_coverage": covered / n,
    }


def best_simpler(results: dict[str, dict]) -> str | None:
    """Highest-accuracy simpler arm among those evaluated; ties go to the earlier in SIMPLER."""
    have = [a for a in SIMPLER if a in results]
    return max(have, key=lambda a: (results[a]["accuracy"], -SIMPLER.index(a))) if have else None


def build_report(results: dict[str, dict], counts: dict[str, int], layer: dict) -> dict:
    return {
        "model": config.MODEL,
        "model_digest": config.MODEL_DIGEST,
        "split_seed": ARMS_SPLIT_SEED,
        "split_counts": counts,
        "scored_on": "eval",
        "layer": layer,
        "results": results,
        "best_simpler_arm": best_simpler(results),
    }


def write_report(path: Path, report: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build(name: str, pool: list[Case], layer: LinearLayer) -> Arm:
    if name == "linear":
        return LinearArm(layer)
    if name in ("prompt", "agent"):
        book = build_label_book(pool)
        if name == "prompt":
            from agent_committee.arms.prompt import make_arm
        else:
            from agent_committee.arms.agent import make_arm
        return make_arm(pool, book, layer)
    raise ValueError(f"unknown arm {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="linear,prompt,agent")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"{args.out} exists; refusing to overwrite", file=sys.stderr)
        return 1
    split = make_arms_split(load_dev("data"))
    layer = make_layer(split.pool)
    decisions = [layer.decide(c.text) for c in split.eval]
    results = {}
    for name in args.arms.split(","):
        results[name] = score(_build(name, split.pool, layer), split.eval, decisions)
        print(name, results[name])
    summary = layer_report(layer, split.eval, decisions)
    write_report(args.out, build_report(results, split.counts(), summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
