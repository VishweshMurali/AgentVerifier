"""Quick test: an exploratory verdict in minutes on a random sample of a scenario's DEV eval split.

    uv run python -m agent_committee.quick --scenario NAME --gap 0.05 [--arms a,b,...] [--yes]

The user chooses the smallest improvement worth paying for (`--gap`). The sample size comes from
the exact McNemar power for that gap at a fixed planning disagreement rate, and the sample is a
fixed-seed random draw, so neither can be re-rolled until the answer changes. The simpler model
the agent must beat is the best free ($0) model on the tune split, chosen before any sampled case
is scored. The plan (gap, size, seed, comparator, arms, base prereg) is written to plan.json and
hashed into the audit before the first model call.

It never opens TEST (no seal import) and every verdict it writes is marked exploratory
(`rehearsal: true`). The sealed run (`runner`) is unchanged and stays fixed-sample. Owner
decisions 2026-11-11: gap chosen by the user with the time shown; comparator = best free model on
tune; planning disagreement 20%; the DEV-score check is not run (it needs the DEV report).
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from agent_committee import checks, power
from agent_committee.arms.config import BANKING77_TASK, MODEL_DIGEST, task_from_dict
from agent_committee.arms.evaluate import CLASSIC
from agent_committee.arms.splits import make_arms_split
from agent_committee.data import load_dev
from agent_committee.prereg import load_prereg
from agent_committee.scenarios import get_scenario
from agent_committee.schemas import Case

PLANNING_DISAGREEMENT = 0.20  # owner decision 2026-11-11 (Banking77 measured 14%)
SAMPLE_SEED = 0  # fixed: a user may not re-draw the sample
AGENT = "agent"
CASCADE = "cascade"
LLM_ARMS = ("prompt", AGENT)
# measured 2026-11-11, llama3.1:8b on an RTX 3060 laptop, uncached: per case, both LLM arms
SECONDS_PER_CASE = {"prompt": 0.29, AGENT: 1.05}
DEV_CHECK_NOTE = (
    "Quick test: the 'score close to its DEV score' check was not run (it needs the DEV report "
    "this test skips); the other checks ran."
)


def plan_size(gap: float, alpha: float, target_power: float) -> int:
    """Paired cases needed to detect `gap` with the exact McNemar test at the planning rate."""
    return power.required_n(gap, PLANNING_DISAGREEMENT, alpha, target_power)


def sample_cases(cases: Sequence[Case], n: int, seed: int = SAMPLE_SEED) -> list[Case]:
    """A fixed-seed simple random sample of `n` cases (all of them when n >= len), by id."""
    ordered = sorted(cases, key=lambda c: c.id)
    if n >= len(ordered):
        return ordered
    picked = np.random.default_rng(seed).choice(len(ordered), size=n, replace=False)
    return sorted((ordered[i] for i in picked), key=lambda c: c.id)


def pick_comparator(free_arms: Mapping[str, object], tune: Sequence[Case]) -> tuple[str, dict]:
    """The free arm with the highest accuracy on tune; ties go to the earlier arm in CLASSIC."""
    scores = {
        name: float(np.mean([arm.predict(c.id, c.text).pred == c.label for c in tune]))
        for name, arm in free_arms.items()
    }
    best = max(scores, key=lambda n: (scores[n], -CLASSIC.index(n)))
    return best, scores


def estimate_minutes(n: int, arms: Sequence[str]) -> float:
    return n * sum(SECONDS_PER_CASE.get(a, 0.0) for a in arms) / 60


def write_plan(path: Path, plan: dict) -> str:
    """plan.json, written before any sampled case is scored; returns its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Quick exploratory test on a DEV sample.")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--gap", type=float, required=True, help="smallest gain worth paying for")
    ap.add_argument("--arms", help="comma-separated arms (default: every runnable arm)")
    ap.add_argument("--prereg", help="base prereg (default: the scenario's, else prereg.yaml)")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--yes", action="store_true", help="run without the size/time confirmation")
    args = ap.parse_args(argv)
    repo = Path(args.repo)

    from agent_committee.runner import RUNNABLE, evaluate, fitted, git_commit, sha256_file

    s = get_scenario(args.scenario, repo)
    if not s.arms_ready:
        print(f"scenario {s.name!r} has no task phrase; the LLM arms cannot run", file=sys.stderr)
        return 1
    names = [a.strip() for a in args.arms.split(",")] if args.arms else list(RUNNABLE)
    unknown = [a for a in names if a not in RUNNABLE]
    free = [a for a in names if a in CLASSIC]
    if unknown or AGENT not in names or not free:
        print(f"arms must be among {list(RUNNABLE)}, include the agent and at least one free "
              f"model; got {names}", file=sys.stderr)  # fmt: skip
        return 1
    base = Path(args.prereg) if args.prereg else repo / s.prereg_path
    if not base.is_file():
        base = repo / "prereg.yaml"
    prereg = load_prereg(base)
    if not 0 < args.gap < 1:
        print("--gap is a fraction of accuracy, e.g. 0.05 for 5 points", file=sys.stderr)
        return 1
    dev = load_dev(repo / s.data_dir)
    split = make_arms_split(dev)
    planned = plan_size(args.gap, prereg.alpha, prereg.power)
    # the power check gates on the planned size for this gap, not the prereg's 2-point size
    prereg = prereg.model_copy(
        update={
            "task": s.name,
            "min_gap": args.gap,
            "checks": prereg.checks.model_copy(update={"min_test_n": planned}),
        }
    )
    # copies of DEV are removed before sampling, so the sample keeps its planned size
    _, clean = checks.leakage(split.eval, split.pool + split.tune, prereg.checks)
    cases = sample_cases(clean, planned)
    llm = [a for a in names if a in LLM_ARMS]
    minutes = estimate_minutes(len(cases), llm)
    deploy_power = power.deploy_rule_power(
        len(cases), args.gap, PLANNING_DISAGREEMENT, prereg.alpha, args.gap, prereg.stats.ci_level
    )
    print(f"gap {args.gap:.3f}: {planned} paired cases planned, {len(cases)} drawn from "
          f"{len(clean)} DEV eval cases (copies of DEV removed)")  # fmt: skip
    if len(cases) < planned:
        print("  fewer cases than planned: expect NEED_MORE_DATA unless the gap is large")
    print(f"  about {minutes:.1f} min of LLM calls if nothing is cached (this laptop)")
    print(f"  chance the full DEPLOY rule fires if the true gain is exactly {args.gap:.3f}: "
          f"{deploy_power:.0%}")  # fmt: skip
    if not args.yes and input("run it? [y/N] ").strip().lower() != "y":
        return 1

    from agent_committee.arms import agent as agent_arm
    from agent_committee.arms import prompt as prompt_arm
    from agent_committee.arms.classic import (
        BoostingArm,
        LinearSVMArm,
        MajorityArm,
        NaiveBayesArm,
        RandomForestArm,
    )
    from agent_committee.arms.labels import build_label_book
    from agent_committee.arms.linear import LinearArm, calibrated_layer
    from agent_committee.llm import LLMClient

    task = task_from_dict(s.task_phrase) if s.task_phrase else BANKING77_TASK
    layer = calibrated_layer(split.pool, split.tune)
    free_factories = {
        "linear": lambda: LinearArm(layer),
        "majority": lambda: fitted(MajorityArm(), split.pool),
        "naive_bayes": lambda: fitted(NaiveBayesArm(), split.pool),
        "linear_svm": lambda: fitted(LinearSVMArm(), split.pool),
        "random_forest": lambda: fitted(RandomForestArm(), split.pool),
        "boosting": lambda: fitted(BoostingArm(), split.pool),
    }
    free_arms = {n: free_factories[n]() for n in free}
    comparator, tune_scores = pick_comparator(free_arms, split.tune)
    print(f"simpler model to beat (best free model on tune): {comparator} {tune_scores}")

    out_dir = repo / s.results_dir / "quick" / f"gap{args.gap:.3f}-n{len(cases)}"
    plan = {
        "scenario": s.name,
        "gap": args.gap,
        "planning_disagreement": PLANNING_DISAGREEMENT,
        "planned_cases": planned,
        "cases": len(cases),
        "sample_seed": SAMPLE_SEED,
        "case_ids_sha256": hashlib.sha256(
            "\n".join(c.id for c in cases).encode("utf-8")
        ).hexdigest(),
        "comparator": comparator,
        "comparator_rule": "best free model on the tune split",
        "tune_accuracy": tune_scores,
        "arms": names,
        "base_prereg": str(base),
        "base_prereg_sha256": sha256_file(base),
        "model_digest": MODEL_DIGEST,
    }
    plan_sha = write_plan(out_dir / "plan.json", plan)
    audit = {
        "git_commit": git_commit(repo),
        "prereg_sha256": sha256_file(base),
        "dev_report_sha256": f"quick test plan {plan_sha}",
        "test_sha256": "quick test: DEV eval sample, sealed test not opened",
        "model_digest": MODEL_DIGEST,
        "layer_fingerprint": layer.fingerprint(),
    }
    report = {"results": {CASCADE: {}}, "best_simpler_arm": comparator}
    book = build_label_book(split.pool)
    with LLMClient(phase="build", model_digest=MODEL_DIGEST) as client:
        arms = dict(free_arms)
        if "prompt" in names:
            arms["prompt"] = prompt_arm.make_arm(split.pool, book, layer, client, task)
        arms[AGENT] = agent_arm.make_arm(split.pool, book, layer, client, task)
        arms = {n: arms[n] for n in names}
        verdict = evaluate(
            prereg, cases, split.pool + split.tune, split.pool, split.tune, layer, arms, report,
            audit, out_dir,
        )  # fmt: skip
    note = (
        f"QUICK TEST on {len(cases)} sampled DEV eval cases (gap {args.gap:.3f}, seed "
        f"{SAMPLE_SEED}): exploratory, not a sealed verdict."
    )
    verdict = verdict.model_copy(
        update={"reasons": [note, DEV_CHECK_NOTE, *verdict.reasons], "rehearsal": True}
    )
    (out_dir / "verdict.json").write_text(verdict.model_dump_json(indent=2) + "\n", "utf-8")
    print(f"verdict: {verdict.verdict.value} -> {out_dir / 'verdict.json'}")
    from agent_committee.report import write_report

    print(f"report: {write_report(verdict, out_dir / 'report.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
