"""Quick test: an exploratory verdict in minutes on a random sample of a scenario's DEV eval split.

    uv run python -m agent_committee.quick --scenario NAME --gap 0.05 [--arms a,b,...] [--yes]

The user chooses the smallest improvement worth paying for (`--gap`). The sample size comes from
the exact McNemar power for that gap at a fixed planning disagreement rate, and the sample is a
fixed-seed random draw, so neither can be re-rolled until the answer changes. The simpler model
the agent must beat is the best free ($0) model on the tune split, chosen before any sampled case
is scored. The plan (gap, size, seed, comparator, arms, base prereg) is written to plan.json and
hashed into the audit before the first model call.

Early stopping (owner approved 2026-11-11): the sample is scored in a fixed shuffled order in 5
equal looks; after each, the agent and the simpler model are compared against the boundaries in
`sequential.py`, and the run stops at the first one crossed. Other arms then run only on the
cases used. A winner needs both the engine and the boundary to agree; the repeated confidence
interval replaces the bootstrap one, and break-even is dropped after an early stop.

Use-case scoping (`--scope`, owner approved 2026-11-11): the user picks the labels their agent
handles. The test mix is every in-scope case, the near misses (labels the linear layer most
often confuses with the scope on the pool) and a fixed-seed slice of everything else making up
20%. A case outside the scope counts right when the model answers any label outside the scope.
Reported per arm: accuracy on in-scope cases and how often other cases are wrongly claimed.

It never opens TEST (no seal import) and every verdict it writes is marked exploratory
(`rehearsal: true`). The sealed run (`runner`) is unchanged and stays fixed-sample. Owner
decisions 2026-11-11: gap chosen by the user with the time shown; comparator = best free model on
tune; planning disagreement 20%; the DEV-score check is not run (it needs the DEV report).
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from agent_committee import checks, power, sequential
from agent_committee.arms.config import BANKING77_TASK, MODEL_DIGEST, task_from_dict
from agent_committee.arms.evaluate import CLASSIC
from agent_committee.arms.splits import make_arms_split
from agent_committee.data import load_dev
from agent_committee.prereg import load_prereg
from agent_committee.scenarios import get_scenario
from agent_committee.schemas import Case, Verdict
from agent_committee.scorers import resolve_scorer

PLANNING_DISAGREEMENT = 0.20  # owner decision 2026-11-11 (Banking77 measured 14%)
SAMPLE_SEED = 0  # fixed: a user may not re-draw the sample
ORDER_SEED = 1  # fixed scoring order for the looks (early stopping, owner approved 2026-11-11)
# use-case scoping (owner approved 2026-11-11): labels picked from a list
NOT_MINE = "not_mine"
OTHERS_SHARE = 0.20  # unrelated labels' share of the scoped test mix
SCOPE_SEED = 2
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


def look_order(cases: Sequence[Case], seed: int = ORDER_SEED) -> list[Case]:
    """The fixed order the sample is scored in. Ids run in label order, so without a shuffle the
    early looks would cover only a few labels."""
    ordered = sorted(cases, key=lambda c: c.id)
    return [ordered[i] for i in np.random.default_rng(seed).permutation(len(ordered))]


def run_looks(
    order: Sequence[Case], agent, simpler, scorer, gap: float, bounds: np.ndarray
) -> "sequential.Look":
    """Score the agent and the simpler model block by block in `order` and stop at the first
    look that crosses a boundary (or at the last look). Other arms run later, only on the cases
    used."""
    d: list[int] = []
    sizes = sequential.look_sizes(len(order), len(bounds))
    done = 0
    for n, c in zip(sizes, bounds, strict=True):
        for case in order[done:n]:
            a = scorer(agent.predict(case.id, case.text).pred, case.label)
            b = scorer(simpler.predict(case.id, case.text).pred, case.label)
            d.append(int(a) - int(b))
        done = n
        look = sequential.check(np.array(d), gap, c)
        print(f"  look at {n} cases: gain {look.diff:+.3f}, z(agent ahead) {look.z0:.2f}, "
              f"z(gain ruled out) {look.z1:.2f}, boundary {c:.2f}")  # fmt: skip
        if look.outcome is not None:
            return look
    return look


def apply_early_stop(verdict, look: "sequential.Look", planned: int):
    """The sequential rule decides; the engine's verdict counts only when it agrees.

    A winner needs both the engine (accuracy, cost, checks) and the boundary crossed in the same
    direction; anything else that is not INVALID becomes NEED_MORE_DATA. The repeated confidence
    interval replaces the bootstrap one (valid at any look); break-even is kept only at a full-size
    run whose repeated interval is entirely above 0.
    """
    if verdict.verdict == Verdict.INVALID:
        return verdict
    lo, hi = look.interval
    agrees = (look.outcome == "agent" and verdict.verdict == Verdict.DEPLOY_AGENT) or (
        look.outcome == "simpler" and verdict.verdict == Verdict.USE_SIMPLER
    )
    reasons = list(verdict.reasons)
    if look.outcome is None:
        reasons.insert(0, "Early stopping: no boundary was crossed by the last look.")
    else:
        side = "the agent is ahead" if look.outcome == "agent" else "a gain of the gap is ruled out"
        reasons.insert(0, f"Early stopping: {side} at {look.n} of {planned} cases.")
    reasons.insert(
        1,
        f"Repeated confidence interval for the gain: {lo:+.3f} to {hi:+.3f}, valid at any look; "
        "an interval quoted below by the engine assumes a fixed-size run.",
    )
    final = verdict.verdict if agrees else Verdict.NEED_MORE_DATA
    if not agrees and verdict.verdict != Verdict.NEED_MORE_DATA:
        reasons.insert(1, f"The engine said {verdict.verdict.value} but the stopping rule did not "
                          "agree, so no winner is picked.")  # fmt: skip
    full = look.n == planned
    keep_break_even = full and lo > 0 and final == Verdict.DEPLOY_AGENT
    if verdict.break_even and not keep_break_even:
        reasons.append("Break-even is not shown: it is not valid after an early stop.")
    return verdict.model_copy(
        update={
            "verdict": final,
            "reasons": reasons,
            "accuracy_diff_ci": (lo, hi),
            "break_even": verdict.break_even if keep_break_even else [],
        }
    )


def near_misses(layer, pool: Sequence[Case], scope: Sequence[str]) -> list[str]:
    """Out-of-scope labels the linear layer most often shortlists for in-scope pool cases (the
    ones most easily confused with the scope), at most as many as the scope has; ties by name.
    Computed on the training pool only, before any test case is scored."""
    inside = set(scope)
    counts: Counter[str] = Counter()
    for c in pool:
        if c.label in inside:
            counts.update(lb for lb in layer.decide(c.text).shortlist if lb not in inside)
    return sorted(counts, key=lambda lb: (-counts[lb], lb))[: len(inside)]


def scoped(
    cases: Sequence[Case], scope: Sequence[str], near: Sequence[str], seed: int = SCOPE_SEED
) -> list[Case]:
    """The scoped mix: every in-scope and near-miss case, plus a fixed-seed random slice of the
    rest sized to OTHERS_SHARE of the mix. Cases outside the scope are relabelled NOT_MINE."""
    inside, close = set(scope), set(near)
    core = [c for c in cases if c.label in inside or c.label in close]
    rest = sorted((c for c in cases if c.label not in inside | close), key=lambda c: c.id)
    k = min(len(rest), round(len(core) * OTHERS_SHARE / (1 - OTHERS_SHARE)))
    picked = np.random.default_rng(seed).choice(len(rest), size=k, replace=False)
    mix = core + [rest[i] for i in picked]
    return sorted(
        (c if c.label in inside else c.model_copy(update={"label": NOT_MINE}) for c in mix),
        key=lambda c: c.id,
    )


class _InScope:
    """An arm whose answers outside the scope read as NOT_MINE. No answer stays no answer (a
    malformed reply still scores wrong)."""

    def __init__(self, arm, scope: Sequence[str]) -> None:
        self._arm, self._scope = arm, set(scope)
        self.name = getattr(arm, "name", None)

    def predict(self, case_id: str, text: str):
        p = self._arm.predict(case_id, text)
        if p.pred is None or p.pred in self._scope:
            return p
        return p.model_copy(update={"pred": NOT_MINE})


def in_scope(arm, scope: Sequence[str]):
    """The arm itself without a scope; with one, its answers mapped to the scope."""
    return _InScope(arm, scope) if scope else arm


def scope_summary(predictions: Path, used: Sequence[Case], scope: Sequence[str]) -> dict:
    """Per arm: accuracy on in-scope cases and the share of other cases claimed as in scope."""
    gold = {c.id: c.label for c in used}
    inside = set(scope)
    tally: dict[str, list[int]] = {}
    for line in predictions.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        t = tally.setdefault(row["arm"], [0, 0, 0, 0])  # in right, in total, grabbed, others
        if gold[row["case_id"]] in inside:
            t[0] += row["correct"]
            t[1] += 1
        else:
            t[2] += row["pred"] in inside
            t[3] += 1
    return {
        arm: {
            "in_scope_cases": t[1],
            "in_scope_accuracy": t[0] / t[1] if t[1] else 0.0,
            "other_cases": t[3],
            "false_grab_rate": t[2] / t[3] if t[3] else 0.0,
        }
        for arm, t in tally.items()
    }


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
    ap.add_argument("--scope", help="comma-separated labels your agent handles (use-case scoping)")
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
        print(f"{s.name!r} has no prereg yet: using {base} (Banking77's ASSUMED prices, error "
              "costs and check thresholds)")  # fmt: skip
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
    n_clean = len(clean)
    from agent_committee.arms.linear import calibrated_layer

    layer = calibrated_layer(split.pool, split.tune)
    scope: list[str] = []
    near: list[str] = []
    pool_ref, tune_ref = split.pool, split.tune
    if args.scope:
        scope = sorted({a.strip() for a in args.scope.split(",") if a.strip()})
        known = {c.label for c in dev}
        wrong = [label for label in scope if label not in known]
        if not scope or wrong or NOT_MINE in known:
            print(f"--scope needs labels of this dataset; unknown: {wrong}", file=sys.stderr)
            return 1
        near = near_misses(layer, split.pool, scope)
        # the test mix, and the references the checks compare it with, are built the same way
        clean = scoped(clean, scope, near)
        pool_ref, tune_ref = scoped(split.pool, scope, near), scoped(split.tune, scope, near)
        print(
            f"scope: {len(scope)} labels, near misses {near}, others {OTHERS_SHARE:.0%} of the mix"
        )
    cases = sample_cases(clean, planned)
    # held-out reference for the drift checks: tune plus the eval cases not drawn (none trained
    # on, none scored); a small reference makes the checks' own false-alarm test noisy
    drawn = {c.id for c in cases}
    held_ref = tune_ref + [c for c in clean if c.id not in drawn]
    llm = [a for a in names if a in LLM_ARMS]
    minutes = estimate_minutes(len(cases), llm)
    deploy_power = power.deploy_rule_power(
        len(cases), args.gap, PLANNING_DISAGREEMENT, prereg.alpha, args.gap, prereg.stats.ci_level
    )
    print(f"gap {args.gap:.3f}: {planned} paired cases planned, {len(cases)} drawn from "
          f"{len(clean)} DEV eval cases (copies of DEV removed)")  # fmt: skip
    if len(cases) < planned:
        print("  fewer cases than planned: expect NEED_MORE_DATA unless the gap is large")
    print(f"  about {minutes:.1f} min of LLM calls if nothing is cached (this laptop), plus "
          f"fitting the free models and the checks (~1-2 min; random forest and boosting add "
          f"~1-3 min more on large datasets)")  # fmt: skip
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
    from agent_committee.arms.linear import LinearArm
    from agent_committee.llm import LLMClient

    task = task_from_dict(s.task_phrase) if s.task_phrase else BANKING77_TASK
    free_factories = {
        "linear": lambda: LinearArm(layer),
        "majority": lambda: fitted(MajorityArm(), split.pool),
        "naive_bayes": lambda: fitted(NaiveBayesArm(), split.pool),
        "linear_svm": lambda: fitted(LinearSVMArm(), split.pool),
        "random_forest": lambda: fitted(RandomForestArm(), split.pool),
        "boosting": lambda: fitted(BoostingArm(), split.pool),
    }
    free_arms = {n: in_scope(free_factories[n](), scope) for n in free}
    comparator, tune_scores = pick_comparator(free_arms, tune_ref)
    print(f"simpler model to beat (best free model on tune): {comparator} {tune_scores}")

    tag = f"-scope{hashlib.sha256(','.join(scope).encode()).hexdigest()[:8]}" if scope else ""
    out_dir = repo / s.results_dir / "quick" / f"gap{args.gap:.3f}-n{len(cases)}{tag}"
    order = look_order(cases)
    bounds = sequential.boundaries(prereg.alpha)
    plan = {
        "early_stopping": {
            "looks": sequential.look_sizes(len(order), len(bounds)),
            "boundaries": [round(float(b), 6) for b in bounds],
            "spending": "Lan-DeMets O'Brien-Fleming, two-sided alpha",
            "order_seed": ORDER_SEED,
            "order_sha256": hashlib.sha256(
                "\n".join(c.id for c in order).encode("utf-8")
            ).hexdigest(),
        },
        "scenario": s.name,
        "gap": args.gap,
        "planning_disagreement": PLANNING_DISAGREEMENT,
        "planned_cases": planned,
        "cases": len(cases),
        "eval_cases": len(split.eval),
        "copies_of_dev_removed": len(split.eval) - n_clean,
        "population": len(clean),
        "scope": {
            "labels": scope,
            "near_misses": near,
            "others_share": OTHERS_SHARE,
            "others_seed": SCOPE_SEED,
            "not_mine_label": NOT_MINE,
        }
        if scope
        else None,
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
    # the cascade (report-only) is derived from linear and the agent: only when both run
    report = {"results": {CASCADE: {}} if "linear" in names else {}, "best_simpler_arm": comparator}
    book = build_label_book(split.pool)
    with LLMClient(phase="build", model_digest=MODEL_DIGEST) as client:
        arms = dict(free_arms)
        if "prompt" in names:
            arms["prompt"] = in_scope(
                prompt_arm.make_arm(split.pool, book, layer, client, task), scope
            )
        arms[AGENT] = in_scope(agent_arm.make_arm(split.pool, book, layer, client, task), scope)
        arms = {n: arms[n] for n in names}
        if prereg.scorer is None:  # no plain-code scorer: the engine returns INVALID unrun
            print("the prereg declares no scorer: INVALID", file=sys.stderr)
            return 1
        scorer = resolve_scorer(prereg.scorer.target)
        look = run_looks(order, arms[AGENT], arms[comparator], scorer, args.gap, bounds)
        used = order[: look.n]
        print(f"stopped at {look.n} of {len(order)} cases: {look.outcome or 'no boundary crossed'}")
        # the sequential design, not the fixed-size power gate, governs the size of this run
        judged = prereg.model_copy(
            update={"checks": prereg.checks.model_copy(update={"min_test_n": look.n})}
        )
        # leakage is checked against all of DEV; drift is compared with references built
        # the same way as the test mix (a scoped mix would otherwise look like drift)
        verdict = evaluate(
            judged, used, split.pool + split.tune, pool_ref, held_ref, layer, arms, report,
            audit, out_dir,
        )  # fmt: skip
    note = (
        f"QUICK TEST on {look.n} of {len(cases)} sampled DEV eval cases (gap {args.gap:.3f}, "
        f"seed {SAMPLE_SEED}): exploratory, not a sealed verdict."
    )
    verdict = apply_early_stop(verdict, look, len(order))
    extra = []
    if scope:
        summary = scope_summary(out_dir / "predictions.jsonl", used, scope)
        (out_dir / "scope.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
        extra = [
            f"Scope: {len(scope)} labels; a case not in scope counts right when the model answers "
            f"any label outside the scope ('{NOT_MINE}')."
        ] + [
            f"{arm}: {v['in_scope_accuracy']:.1%} right on in-scope cases, "
            f"{v['false_grab_rate']:.1%} of other cases wrongly claimed as in scope"
            for arm, v in summary.items()
        ]
    verdict = verdict.model_copy(
        update={"reasons": [note, DEV_CHECK_NOTE, *extra, *verdict.reasons], "rehearsal": True}
    )
    (out_dir / "verdict.json").write_text(verdict.model_dump_json(indent=2) + "\n", "utf-8")
    print(f"verdict: {verdict.verdict.value} -> {out_dir / 'verdict.json'}")
    from agent_committee.report import write_report

    print(f"report: {write_report(verdict, out_dir / 'report.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
