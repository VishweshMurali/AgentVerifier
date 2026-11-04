import math

import numpy as np
import pytest

from agent_committee import decision
from agent_committee.decision import ArmRun, break_even_cost, decide, inference_cost
from agent_committee.prereg import load_prereg
from agent_committee.schemas import (
    AuditBundle,
    CheckEffect,
    CheckResult,
    EvidenceLevel,
    PriceTable,
    Verdict,
)

BASE = load_prereg()
PREREG = BASE.model_copy(
    update={
        "stats": BASE.stats.model_copy(update={"bootstrap_replicates": 500, "aa_simulations": 200})
    }
)
AUDIT = AuditBundle(
    git_commit="c",
    prereg_sha256="p",
    dev_report_sha256="d",
    test_sha256="t",
    model_digest="m",
    layer_fingerprint="f",
    predictions_sha256="x",
)


def arm(name, accuracy, n=3000, prompt=0.0, completion=0.0, seed=0):
    rng = np.random.default_rng(seed)
    correct = rng.random(n) < accuracy
    return ArmRun(name, correct, np.full(n, prompt), np.full(n, completion), np.full(n, 0.5))


def run(agent, simpler, **kw):
    args = dict(
        others=[],
        flagged=np.zeros(len(agent.correct), dtype=bool),
        prereg=PREREG,
        checks=[],
        scorer="exact_label",
        floor_passed=True,
        evidence=EvidenceLevel.REAL,
        audit=AUDIT,
        n_dropped=0,
    )
    args.update(kw)
    return decide(agent, simpler, **args)


def test_inference_cost_per_case():
    table = PriceTable(name="t", input_usd_per_mtok=1.0, output_usd_per_mtok=2.0)
    assert inference_cost(1000, 500, table) == pytest.approx(0.002)


def test_break_even_cost():
    assert break_even_cost(0.01, 0.10, 0.0, 0.20) == pytest.approx(0.1)
    assert break_even_cost(0.0, 0.10, 0.01, 0.20) == 0.0
    assert math.isinf(break_even_cost(0.01, 0.20, 0.0, 0.10))


def test_worse_and_dearer_agent_means_use_simpler():
    v = run(arm("agent", 0.82, prompt=700, completion=26, seed=1), arm("linear", 0.90, seed=2))
    assert v.verdict == Verdict.USE_SIMPLER
    assert v.break_even == [] and v.accuracy_diff < 0
    assert v.mcnemar_counts is not None and v.n_cases == 3000


def test_clearly_better_agent_is_deployed_with_break_even():
    v = run(arm("agent", 0.96, prompt=700, completion=26, seed=1), arm("linear", 0.86, seed=2))
    assert v.verdict == Verdict.DEPLOY_AGENT
    assert len(v.break_even) == 3
    for b in v.break_even:
        assert 0 < b.ci_low <= b.value <= b.ci_high


def test_too_few_cases_means_need_more_data():
    agent = arm("agent", 0.9, n=80, prompt=700, seed=1)
    v = run(agent, arm("linear", 0.9, n=80, seed=2))
    assert v.verdict == Verdict.NEED_MORE_DATA


def test_failed_floor_or_missing_scorer_is_invalid():
    a, s = arm("agent", 0.96, seed=1), arm("linear", 0.86, seed=2)
    assert run(a, s, floor_passed=False).verdict == Verdict.INVALID
    assert run(a, s, scorer=None).verdict == Verdict.INVALID


def test_failed_checks_bind_the_verdict():
    a, s = arm("agent", 0.96, seed=1), arm("linear", 0.86, seed=2)
    nmd = CheckResult(name="power_cap", value=1, passed=False, effect=CheckEffect.NEED_MORE_DATA)
    inv = CheckResult(name="score_drift", value=1, passed=False, effect=CheckEffect.INVALID)
    rep = CheckResult(name="label_mix", value=1, passed=False, effect=CheckEffect.REPORT)
    assert run(a, s, checks=[nmd]).verdict == Verdict.NEED_MORE_DATA
    assert run(a, s, checks=[inv, nmd]).verdict == Verdict.INVALID
    assert run(a, s, checks=[rep]).verdict == Verdict.DEPLOY_AGENT


def test_synthetic_evidence_never_picks_a_winner():
    a, s = arm("agent", 0.96, seed=1), arm("linear", 0.86, seed=2)
    v = run(a, s, evidence=EvidenceLevel.SYNTHETIC)
    assert v.verdict == Verdict.NEED_MORE_DATA and v.accuracy


def test_arm_results_carry_costs_and_flagged_accuracy():
    a, s = arm("agent", 0.8, prompt=700, completion=26, seed=1), arm("linear", 0.9, seed=2)
    flagged = np.arange(3000) % 10 == 0
    v = run(a, s, flagged=flagged, others=[arm("prompt", 0.75, prompt=230, seed=3)])
    names = [r.name for r in v.arms]
    assert names == ["agent", "linear", "prompt"]
    agent_result = v.arms[0]
    assert set(agent_result.inference_cost) == {t.name for t in PREREG.price_tables}
    assert agent_result.inference_cost["cloud_small"] > 0
    assert v.arms[1].inference_cost["cloud_small"] == 0
    assert agent_result.flagged_accuracy is not None
    assert v.flagged_rate == pytest.approx(0.1)


def test_decision_never_reads_latency():
    a, s = arm("agent", 0.82, prompt=700, seed=1), arm("linear", 0.90, seed=2)
    slow = decision.ArmRun(
        a.name, a.correct, a.prompt_tokens, a.completion_tokens, a.latency_s * 100
    )
    assert run(slow, s).verdict == run(a, s).verdict
