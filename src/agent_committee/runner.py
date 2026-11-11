"""Phase D runner: the ONLY module that opens the sealed TEST set.

    uv run python -m agent_committee.runner              # the one sealed run (needs arms-frozen)
    uv run python -m agent_committee.runner --rehearse   # same pipeline on the DEV eval split

Order: unseal guard -> drop test texts copied from DEV (leakage) -> every arm answers the same
cases -> checks -> exact McNemar, case bootstrap, A/A -> decision engine -> results/verdict.json,
predictions and the HTML report. The rehearsal never touches TEST: it stands the DEV eval split in
for it, reuses the build-phase LLM cache and writes to results/rehearsal/.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent_committee import checks
from agent_committee.arms.cascade import cascade_predictions
from agent_committee.arms.config import BANKING77_TASK, MODEL_DIGEST, task_from_dict
from agent_committee.arms.splits import make_arms_split
from agent_committee.data import load_dev
from agent_committee.decision import ArmRun, decide
from agent_committee.prereg import load_prereg
from agent_committee.scenarios import BANKING77, get_scenario
from agent_committee.schemas import (
    AuditBundle,
    Case,
    CheckEffect,
    CheckResult,
    EvidenceLevel,
    Prereg,
    Verdict,
    VerdictJSON,
)
from agent_committee.scorers import resolve_scorer

DEV_REPORT = BANKING77.dev_report
FALSE_ALARM_REPS = 100
RUN_MARKER = "sealed_run.started"  # written right after unsealing; blocks any second sealed run
AGENT = "agent"
CASCADE = "cascade"
PROGRESS_EVERY = 200


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PairedRun:
    """Every arm's outcome on the same cases, plus the agent's behaviour for the quality floor."""

    runs: dict[str, ArmRun]
    rows: list[dict]
    own_tool_rate: float
    send_back_rate: float


def run_arms(
    arms: Mapping[str, object],
    cases: list[Case],
    scorer: Callable[[str | None, str], bool],
    log: Callable[[str], None] = print,
    cascade_flags: Sequence[bool] | None = None,
) -> PairedRun:
    """Each arm answers every case in the same order. Gold labels reach only the scorer.

    With `cascade_flags` (the layer's Null flags on `cases`), the cascade is derived from the
    linear and agent predictions after both ran (it makes no model calls of its own).
    """
    made: dict[str, list] = {}
    for name, arm in arms.items():
        preds = []
        for i, c in enumerate(cases, 1):
            preds.append(arm.predict(c.id, c.text))
            if i % PROGRESS_EVERY == 0:
                log(f"{name}: {i}/{len(cases)}")
        made[name] = preds
    if cascade_flags is not None:
        made[CASCADE] = cascade_predictions(made["linear"], made[AGENT], cascade_flags)
    runs, rows = {}, []
    own, back = 0.0, 0.0
    for name, preds in made.items():
        correct = np.array([scorer(p.pred, c.label) for p, c in zip(preds, cases, strict=True)])
        runs[name] = ArmRun(
            name,
            correct,
            np.array([p.prompt_tokens for p in preds], dtype=float),
            np.array([p.completion_tokens for p in preds], dtype=float),
            np.array([p.latency_s for p in preds], dtype=float),
        )
        rows += [
            {
                "case_id": p.case_id,
                "arm": name,
                "pred": p.pred,
                "correct": bool(ok),
                "prompt_tokens": p.prompt_tokens,
                "completion_tokens": p.completion_tokens,
                "latency_s": p.latency_s,
                "own_tool_calls": p.own_tool_calls,
                "sent_back": p.sent_back,
                "malformed": p.malformed,
                "too_long": p.too_long,
            }
            for p, ok in zip(preds, correct, strict=True)
        ]
        if name == AGENT:
            own = float(np.mean([p.own_tool_calls > 0 and not p.sent_back for p in preds]))
            back = float(np.mean([p.sent_back for p in preds]))
        log(f"{name}: done, accuracy {correct.mean():.4f}")
    return PairedRun(runs, rows, own, back)


def floor_check(own_tool_rate: float, send_back_rate: float, prereg: Prereg) -> CheckResult:
    floor = prereg.quality_floor
    passed = own_tool_rate >= floor.min_own_tool_rate and send_back_rate <= floor.max_send_back_rate
    return CheckResult(
        name="quality_floor",
        value=own_tool_rate,
        low=floor.min_own_tool_rate,
        passed=passed,
        effect=CheckEffect.INVALID,
        detail=f"own tool use {own_tool_rate:.3f}, send-backs {send_back_rate:.3f}",
    )


def build_checks(
    prereg: Prereg,
    paired: PairedRun,
    cases: list[Case],
    decisions: list,
    leak: CheckResult,
    pool: list[Case],
    held_out: list[Case],
    held_decisions: list,
    dev_results: Mapping[str, Mapping],
    simpler: str,
    false_alarm_reps: int = FALSE_ALARM_REPS,
) -> list[CheckResult]:
    """Every pre-registered check, run on the paired cases against DEV references."""
    cfg, st, own = prereg.checks, prereg.stats, True
    agent_ok, simpler_ok = paired.runs[AGENT].correct, paired.runs[simpler].correct
    out = [leak, checks.power_cap(len(cases), cfg)]
    out.append(floor_check(paired.own_tool_rate, paired.send_back_rate, prereg))
    for name, run in paired.runs.items():
        ref = dev_results.get(name)
        if ref:
            out.append(
                checks.tune_gap(
                    name, ref["correct"], ref["n"], int(run.correct.sum()), len(cases), cfg, "dev"
                )
            )
    flagged = [d.null for d in decisions]
    out.append(checks.null_rate(sum(flagged), len(cases), cfg))
    out.append(
        checks.score_drift(
            [d.score for d in held_decisions], [d.score for d in decisions], cfg, own
        )
    )
    out.append(
        checks.domain_auc([c.text for c in pool], [c.text for c in cases], cfg, own, st.seed)
    )
    out.append(checks.label_mix([c.label for c in pool], [c.label for c in cases], cfg))
    out.append(
        checks.false_alarm_rate(
            [c.text for c in held_out],
            [d.score for d in held_decisions],
            cfg,
            false_alarm_reps,
            st.seed,
        )
    )
    labels = [c.label for c in cases]
    out.append(checks.per_label_bias(agent_ok, simpler_ok, labels, cfg, prereg.alpha, st.seed))
    out.append(
        checks.flagged_split(
            agent_ok, simpler_ok, flagged, st.bootstrap_replicates, st.ci_level, st.seed
        )
    )
    return out


RUNNABLE = (
    "majority",
    "linear",
    "naive_bayes",
    "linear_svm",
    "random_forest",
    "boosting",
    "prompt",
    AGENT,
)


def arms_to_run(dev_results: Mapping[str, object]) -> list[str] | None:
    """The arms a scenario's frozen DEV report scored, minus the derived cascade, in ladder order
    (RUNNABLE; Banking77 v1 runs linear, prompt, agent as before). None when the report cannot be
    run: an unknown arm, no agent, or a cascade without linear."""
    if set(dev_results) - {*RUNNABLE, CASCADE}:
        return None
    names = [n for n in RUNNABLE if n in dev_results]
    if AGENT not in names:
        return None
    if CASCADE in dev_results and "linear" not in names:
        return None
    return names


def fitted(arm, pool: list[Case]):
    arm.fit(pool)
    return arm


def sealed_run_traces(out_dir: Path) -> list[Path]:
    """Files left by an earlier sealed run. Any of them means the one sealed run already began."""
    names = ("verdict.json", "predictions.jsonl", RUN_MARKER)
    return [out_dir / n for n in names if (out_dir / n).exists()]


def git_commit(repo: Path) -> str:
    done = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def write_predictions(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def evaluate(
    prereg: Prereg,
    cases: list[Case],
    reference: list[Case],
    pool: list[Case],
    held_out: list[Case],
    layer,
    arms: Mapping[str, object],
    dev_report: Mapping,
    audit: dict,
    out_dir: Path,
    false_alarm_reps: int = FALSE_ALARM_REPS,
    log: Callable[[str], None] = print,
) -> VerdictJSON:
    """Leakage, the paired run, checks and the decision. Writes predictions.jsonl in out_dir.

    A prereg without a scorer, quality floor or checks cannot be judged: that is INVALID, decided
    before any arm runs.
    """
    missing = [n for n in ("scorer", "quality_floor", "checks") if getattr(prereg, n) is None]
    if missing:
        return VerdictJSON(
            verdict=Verdict.INVALID,
            evidence_level=EvidenceLevel.REAL,
            audit=AuditBundle(**audit),
            reasons=[f"prereg.yaml does not set {', '.join(missing)}; nothing was run."],
            scorer=prereg.scorer.name if prereg.scorer else None,
            accuracy={},
        )
    leak, kept = checks.leakage(cases, reference, prereg.checks)
    log(f"{len(kept)} cases kept, {len(cases) - len(kept)} dropped as copies of DEV")
    scorer = resolve_scorer(prereg.scorer.target)
    decisions = [layer.decide(c.text) for c in kept]
    # the cascade runs only where the DEV report has it (Banking77's v1 report does not)
    flags = [d.null for d in decisions] if CASCADE in dev_report["results"] else None
    paired = run_arms(arms, kept, scorer, log, cascade_flags=flags)
    predictions_sha = write_predictions(paired.rows, out_dir / "predictions.jsonl")
    held_decisions = [layer.decide(c.text) for c in held_out]
    simpler = dev_report["best_simpler_arm"]
    found = build_checks(
        prereg,
        paired,
        kept,
        decisions,
        leak,
        pool,
        held_out,
        held_decisions,
        dev_report["results"],
        simpler,
        false_alarm_reps,
    )
    floor = next(c for c in found if c.name == "quality_floor")
    others = [r for n, r in paired.runs.items() if n not in (AGENT, simpler)]
    coverage = float(
        np.mean([c.label in d.shortlist for c, d in zip(kept, decisions, strict=True)])
    )
    return decide(
        paired.runs[AGENT],
        paired.runs[simpler],
        others,
        np.array([d.null for d in decisions], dtype=bool),
        prereg,
        found,
        prereg.scorer.name,
        floor.passed,
        EvidenceLevel.REAL,
        AuditBundle(**audit, predictions_sha256=predictions_sha),
        len(cases) - len(kept),
        coverage,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase D: the sealed run (or a rehearsal on DEV).")
    ap.add_argument("--rehearse", action="store_true", help="run on the DEV eval split instead")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--scenario", default=BANKING77.name)
    args = ap.parse_args(argv)
    repo = Path(args.repo)
    s = get_scenario(args.scenario, repo)
    if not s.arms_ready:
        # the arms would run on another scenario's null threshold, shortlist and DEV report
        print(f"scenario {s.name!r} has no task phrase in its scenario.yaml; refusing",
              file=sys.stderr)  # fmt: skip
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
    from agent_committee.arms.linear import LinearArm, make_layer
    from agent_committee.llm import LLMClient

    prereg = load_prereg(repo / s.prereg_path, task=s.name)
    if prereg.linear is None or not (repo / s.dev_report).is_file():
        # checked before anything is unsealed
        print(f"{s.name!r} needs prereg.linear and its DEV report {s.dev_report}", file=sys.stderr)
        return 1
    dev_report = json.loads((repo / s.dev_report).read_text(encoding="utf-8"))
    task = task_from_dict(s.task_phrase) if s.task_phrase else BANKING77_TASK
    # Banking77's v1 report predates the field: it was scored with the Banking77 phrase
    reported = dev_report.get("task_phrase", BANKING77_TASK.as_dict())
    if reported != task.as_dict():
        print(f"the DEV report was scored with task phrase {reported}, not {task.as_dict()}",
              file=sys.stderr)  # fmt: skip
        return 1
    dev = load_dev(repo / s.data_dir)
    names = arms_to_run(dev_report["results"])
    if names is None:
        # checked before anything is unsealed
        print(f"DEV report arms cannot be run: {sorted(dev_report['results'])}", file=sys.stderr)
        return 1
    split = make_arms_split(dev)
    # the pre-registered threshold and shortlist (Banking77: the v1 values, so the v1 fingerprint)
    layer = make_layer(split.pool, prereg.linear.null_threshold, prereg.linear.shortlist_k)
    if layer.fingerprint() != dev_report["layer"]["fingerprint"]:
        print("the rebuilt linear layer does not match the DEV report", file=sys.stderr)
        return 1

    if args.rehearse:
        # a rehearsal may be repeated; it never opens TEST and never claims the test hash
        out_dir = repo / s.results_dir / "rehearsal"
        cases, reference, held_out, phase = split.eval, split.pool + split.tune, split.tune, "build"
        test_sha = "rehearsal: DEV eval split, sealed test not opened"
    else:
        # the sealed run happens once: refuse before unsealing if any trace of a run exists
        out_dir = repo / s.results_dir
        traces = sealed_run_traces(out_dir)
        if traces:
            found = ", ".join(str(p) for p in traces)
            print(f"a sealed run already started ({found}); refusing", file=sys.stderr)
            return 1
        from agent_committee import seal

        cases = seal.load_test(repo, s=s)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / RUN_MARKER).write_text(git_commit(repo) + "\n", encoding="utf-8")
        reference, held_out, phase = list(dev), split.tune + split.eval, "run"
        test_sha = (repo / s.data_dir / "test.sha256").read_text(encoding="utf-8").strip()
    verdict_path = out_dir / "verdict.json"

    audit = {
        "git_commit": git_commit(repo),
        "prereg_sha256": sha256_file(repo / s.prereg_path),
        "dev_report_sha256": sha256_file(repo / s.dev_report),
        "test_sha256": test_sha,
        "model_digest": MODEL_DIGEST,
        "layer_fingerprint": layer.fingerprint(),
    }
    book = build_label_book(split.pool)
    with LLMClient(phase=phase, model_digest=MODEL_DIGEST) as client:
        # exactly the arms the scenario's frozen DEV report scored (the cascade is derived)
        factories = {
            "linear": lambda: LinearArm(layer),
            "prompt": lambda: prompt_arm.make_arm(split.pool, book, layer, client, task),
            AGENT: lambda: agent_arm.make_arm(split.pool, book, layer, client, task),
            "majority": lambda: fitted(MajorityArm(), split.pool),
            "naive_bayes": lambda: fitted(NaiveBayesArm(), split.pool),
            "linear_svm": lambda: fitted(LinearSVMArm(), split.pool),
            "random_forest": lambda: fitted(RandomForestArm(), split.pool),
            "boosting": lambda: fitted(BoostingArm(), split.pool),
        }
        arms = {n: factories[n]() for n in names}
        verdict = evaluate(
            prereg, cases, reference, split.pool, held_out, layer, arms, dev_report, audit, out_dir
        )
    if args.rehearse:
        note = "REHEARSAL on the DEV eval split, not the sealed test: this is not a verdict."
        verdict = verdict.model_copy(
            update={"reasons": [note, *verdict.reasons], "rehearsal": True}
        )
    verdict_path.write_text(verdict.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"verdict: {verdict.verdict.value} -> {verdict_path}")
    try:
        from agent_committee.report import write_report
    except ImportError:
        print("report module missing; verdict.json written without report.html", file=sys.stderr)
    else:
        print(f"report: {write_report(verdict, out_dir / 'report.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
