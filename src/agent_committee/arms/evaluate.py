"""Score arms on the DEV eval split only and write arms/dev_report.json.

    python -m agent_committee.arms.evaluate [--arms NAMES] [--out PATH] [--data-dir DIR]
        [--task-item S --task-items S --task-label-kind S [--task-hint S]]

Default arms (DEFAULT_ARMS): majority, linear, naive_bayes, linear_svm, random_forest, boosting
(no LLM, CLASSIC), then prompt and agent. With both
linear and agent, the cascade (linear on confident cases, agent on Null-flagged ones) is added
from their predictions.

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
from agent_committee.arms.cascade import cascade_predictions
from agent_committee.arms.classic import (
    BoostingArm,
    LinearSVMArm,
    MajorityArm,
    NaiveBayesArm,
    RandomForestArm,
)
from agent_committee.arms.labels import build_label_book
from agent_committee.arms.linear import Decision, LinearArm, LinearLayer, calibrated_layer
from agent_committee.arms.splits import ARMS_SPLIT_SEED, make_arms_split
from agent_committee.data import load_dev
from agent_committee.schemas import ArmPrediction, Case
from agent_committee.scorers import Scorer, get_scorer

REPORT_PATH = Path(__file__).with_name("dev_report.json")
# the no-LLM rungs: tier 1 (majority floor, linear, Naive Bayes, linear SVM), tier 2 (forest,
# boosting); they run on any dataset, task phrase or not
CLASSIC = ("majority", "linear", "naive_bayes", "linear_svm", "random_forest", "boosting")
# arms the agent has to beat, simplest first (ties go to the earlier one, so a free arm beats a
# paid one on a tie); the cascade is not one
SIMPLER = (*CLASSIC, "prompt")
CLASSIC_FIT = {
    "majority": MajorityArm,
    "naive_bayes": NaiveBayesArm,
    "linear_svm": LinearSVMArm,
    "random_forest": RandomForestArm,
    "boosting": BoostingArm,
}
DEFAULT_ARMS = ",".join(
    [*SIMPLER, "agent"]
)  # the whole ladder; the cascade is added from linear + agent


def score(
    arm: Arm, cases: list[Case], decisions: list[Decision], scorer: Scorer | None = None
) -> dict:
    """One arm's results. `decisions` are the shared layer's, aligned with `cases`."""
    return summarize([arm.predict(c.id, c.text) for c in cases], cases, decisions, scorer)


def summarize(
    preds: list[ArmPrediction],
    cases: list[Case],
    decisions: list[Decision],
    scorer: Scorer | None = None,
) -> dict:
    """Results of predictions already made, aligned with `cases` and `decisions`."""
    scorer = scorer or get_scorer("exact_label")
    right = [scorer(p.pred, c.label) for p, c in zip(preds, cases, strict=True)]
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


def _build(
    name: str, pool: list[Case], layer: LinearLayer, task: config.TaskPhrase = config.BANKING77_TASK
) -> Arm:
    if name == "linear":
        return LinearArm(layer)
    if name in CLASSIC_FIT:
        arm = CLASSIC_FIT[name]()
        arm.fit(pool)
        return arm
    if name in ("prompt", "agent"):
        book = build_label_book(pool)
        if name == "prompt":
            from agent_committee.arms.prompt import make_arm
        else:
            from agent_committee.arms.agent import make_arm
        return make_arm(pool, book, layer, task=task)
    raise ValueError(f"unknown arm {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None, help="a scenario's DEV folder")
    # a scenario by name: its DEV folder, report path and task phrase come from its files, so no
    # task text passes through a shell (C9-Q1); the form check_dataset prints
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--shortlist-k", type=int, default=config.SHORTLIST_K)
    # the scenario's task phrase, as plain strings (JSON on a Windows command line loses quotes);
    # check_dataset prints the exact command with these filled from scenario.yaml
    ap.add_argument("--task-item", default=None)
    ap.add_argument("--task-items", default=None)
    ap.add_argument("--task-label-kind", default=None)
    ap.add_argument("--task-hint", default=None)
    args = ap.parse_args(argv)
    if args.scenario is not None:
        manual = (args.data_dir, args.out, args.task_item, args.task_items)
        if any(v is not None for v in (*manual, args.task_label_kind, args.task_hint)):
            print("--scenario takes its folder, report and task phrase from the scenario's "
                  "files: drop --data-dir, --out and --task-*", file=sys.stderr)  # fmt: skip
            return 1
        from agent_committee.scenarios import get_scenario

        try:
            s = get_scenario(args.scenario, args.repo)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        args.data_dir, args.out = args.repo / s.data_dir, args.repo / s.dev_report
        # Banking77 (built in) has no scenario.yaml phrase: it is the v1 default
        phrase = s.task_phrase or (config.BANKING77_TASK.as_dict() if s.name == "banking77" else {})
        for key, value in phrase.items():
            setattr(args, f"task_{key}", value)
    if args.data_dir is not None and args.out is None:
        # another scenario's report must never land on Banking77's
        print("--data-dir needs --out (the scenario's own DEV report path)", file=sys.stderr)
        return 1
    llm = sorted({"prompt", "agent"} & set(args.arms.split(",")))
    given = {
        k: v
        for k, v in zip(
            config.TASK_KEYS,
            (args.task_item, args.task_items, args.task_label_kind, args.task_hint),
            strict=True,
        )
        if v is not None
    }
    try:
        task = config.task_from_dict(given) if given else None
    except ValueError as exc:
        print(f"--task-*: {exc}", file=sys.stderr)
        return 1
    if args.data_dir is not None and llm and task is None:
        # without the scenario's own phrase the LLM arms would be told about banking queries
        print(
            f"{llm} need the scenario's task phrase on another dataset: pass --task-item, "
            "--task-items and --task-label-kind (check_dataset prints the command), or run "
            f"only --arms {','.join(CLASSIC)}",
            file=sys.stderr,
        )
        return 1
    if task is None and args.data_dir is None:
        task = config.BANKING77_TASK
    out = args.out or REPORT_PATH
    if out.exists():
        print(f"{out} exists; refusing to overwrite", file=sys.stderr)
        return 1
    split = make_arms_split(load_dev(args.data_dir or "data"))
    # the Null threshold is always the tune 10th percentile (0.486288 on Banking77)
    layer = calibrated_layer(split.pool, split.tune, args.shortlist_k)
    decisions = [layer.decide(c.text) for c in split.eval]
    results, preds = {}, {}
    for name in args.arms.split(","):
        arm = _build(name, split.pool, layer, task or config.BANKING77_TASK)
        preds[name] = [arm.predict(c.id, c.text) for c in split.eval]
        results[name] = summarize(preds[name], split.eval, decisions)
        print(name, results[name])
    if "linear" in preds and "agent" in preds:
        flagged = [d.null for d in decisions]
        cascade = cascade_predictions(preds["linear"], preds["agent"], flagged)
        results["cascade"] = summarize(cascade, split.eval, decisions)
        print("cascade", results["cascade"])
    summary = layer_report(layer, split.eval, decisions)
    report = build_report(results, split.counts(), summary)
    # the runner refuses a scenario whose phrase differs; a non-LLM preview of another dataset
    # records none (it never told a model anything)
    report["task_phrase"] = task.as_dict() if task else None
    write_report(out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
