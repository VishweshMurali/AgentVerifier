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
        "stats": BASE.stats.model_copy(update={"bootstrap_replicates": 500, "aa_simulations": 100})
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


# Ties and edges of the frozen rules (pinned as they are under arms-frozen; see R9 in the review).


def paired(agent_only, simpler_only, n=3000):
    """Agent and simpler arms with exact discordant counts; the agent costs more per case."""
    agent_ok = np.ones(n, dtype=bool)
    simpler_ok = np.ones(n, dtype=bool)
    agent_ok[:simpler_only] = False
    simpler_ok[simpler_only : simpler_only + agent_only] = False
    tokens = np.full(n, 700.0)
    agent = ArmRun("agent", agent_ok, tokens, np.full(n, 26.0), np.full(n, 0.5))
    simpler = ArmRun("linear", simpler_ok, np.zeros(n), np.zeros(n), np.full(n, 0.5))
    return agent, simpler


def test_a_gain_exactly_at_min_gap_deploys():
    v = run(*paired(100, 40))
    assert v.accuracy_diff == PREREG.min_gap
    assert v.verdict == Verdict.DEPLOY_AGENT and len(v.break_even) == 3


def test_a_gain_just_below_min_gap_does_not_deploy():
    v = run(*paired(99, 40))
    assert v.accuracy_diff < PREREG.min_gap and v.accuracy_diff_ci[0] > 0
    assert v.verdict == Verdict.NEED_MORE_DATA and v.break_even == []
    assert any("below the 0.02 minimum gap" in r for r in v.reasons)


def test_mcnemar_p_equal_to_alpha_does_not_deploy(monkeypatch):
    monkeypatch.setattr(decision, "mcnemar_exact", lambda a, b: (100, 40, PREREG.alpha))
    v = run(*paired(100, 40))
    assert v.verdict == Verdict.NEED_MORE_DATA and v.break_even == []
    assert any("McNemar p=0.05 is not below alpha=0.05" in r for r in v.reasons)


def test_ci_lower_bound_equal_to_zero_does_not_deploy(monkeypatch):
    monkeypatch.setattr(decision, "percentile_ci", lambda samples, level: (0.0, 0.05))
    v = run(*paired(100, 40))
    assert v.verdict == Verdict.NEED_MORE_DATA and v.break_even == []
    assert any("CI is not entirely above 0" in r for r in v.reasons)


def test_ci_upper_bound_equal_to_min_gap_is_not_use_simpler(monkeypatch):
    monkeypatch.setattr(decision, "percentile_ci", lambda samples, level: (-0.05, PREREG.min_gap))
    v = run(*paired(40, 100))
    assert v.verdict == Verdict.NEED_MORE_DATA
    assert any("still allows a gain" in r for r in v.reasons)


def test_arms_of_unequal_length_are_refused():
    agent, _ = paired(100, 40)
    _, simpler = paired(100, 40, n=2999)
    with pytest.raises(ValueError):
        run(agent, simpler)


def test_no_break_even_cause_names_the_first_unmet_condition():
    cause = decision._no_break_even_cause
    assert cause(0.001, 0.0, 0.05, PREREG) == "the accuracy difference CI is not entirely above 0"
    assert cause(0.05, 0.01, 0.05, PREREG) == "McNemar p=0.05 is not below alpha=0.05"
    assert cause(0.001, 0.01, 0.019, PREREG) == "the gain +0.019 is below the 0.02 minimum gap"
    assert cause(0.001, 0.01, 0.05, PREREG) == (
        "a pre-registered check or rule stopped the decision before it"
    )


def test_every_verdict_but_deploy_states_why_there_is_no_break_even():
    v = run(*paired(40, 100))
    assert v.verdict == Verdict.USE_SIMPLER
    assert v.reasons[-1] == (
        "No break-even error cost: the accuracy difference CI is not entirely above 0."
    )
    assert not any(r.startswith("No break-even") for r in run(*paired(100, 40)).reasons)


def test_synthetic_checked_evidence_can_deploy():
    v = run(*paired(100, 40), evidence=EvidenceLevel.SYNTHETIC_CHECKED)
    assert v.verdict == Verdict.DEPLOY_AGENT


# P1a: cost-aware DEPLOY and USE_SIMPLER, A/A on the whole rule, paired lengths.


def costed(agent_only, simpler_only, agent_tokens, simpler_tokens, n=3000):
    agent, simpler = paired(agent_only, simpler_only, n)
    return (
        ArmRun("agent", agent.correct, np.full(n, agent_tokens), np.zeros(n), agent.latency_s),
        ArmRun(
            "linear", simpler.correct, np.full(n, simpler_tokens), np.zeros(n), simpler.latency_s
        ),
    )


def test_a_significant_gain_is_not_deployed_if_the_agent_is_never_cheaper():
    # +2.0 points of accuracy but the agent costs so much per case that no (price table, error
    # cost) scenario favours it: E[C] gap = 0.02 * 5 - 5000 tokens * $/token is positive.
    agent, simpler = costed(100, 40, 1e9, 0)
    v = run(agent, simpler)
    assert v.accuracy_diff == PREREG.min_gap and v.accuracy_diff_ci[0] > 0
    assert v.verdict != Verdict.DEPLOY_AGENT and v.break_even == []
    assert any("no lower expected cost" in r for r in v.reasons)


def test_the_verdict_records_which_scenarios_favour_each_arm():
    v = run(*paired(100, 40))
    line = next(r for r in v.reasons if r.startswith("Lower expected cost per scenario"))
    assert "agent:" in line and "linear:" in line and "@$" in line


def test_simpler_that_costs_more_but_is_far_more_accurate_is_use_simpler():
    # R2: agent 0.80 at 100 tokens, simpler 0.92 at 300 tokens; simpler wins every scenario.
    n = 3000
    rng = np.random.default_rng(0)
    simpler_ok = rng.random(n) < 0.92
    agent_ok = simpler_ok & (rng.random(n) < 0.87)
    agent = ArmRun("agent", agent_ok, np.full(n, 100.0), np.zeros(n), np.full(n, 0.5))
    simpler = ArmRun("linear", simpler_ok, np.full(n, 300.0), np.zeros(n), np.full(n, 0.5))
    v = run(agent, simpler)
    assert v.verdict == Verdict.USE_SIMPLER
    assert any("lower expected cost in every price table" in r for r in v.reasons)


def test_aa_runs_the_whole_decide_rule(monkeypatch):
    calls = []
    real = decision.decide

    def spy(*args, **kw):
        calls.append(kw.get("run_aa"))
        return real(*args, **kw)

    monkeypatch.setattr(decision, "decide", spy)
    v = run(*paired(100, 40))
    assert v.aa_false_winner_rate is not None
    assert calls.count(False) >= PREREG.stats.aa_simulations  # decide was called per simulation


def test_aa_rate_moves_with_the_winner_rule(monkeypatch):
    agent, simpler = paired(100, 40)
    real = decision.VerdictJSON

    def always_deploys(*args, **kw):
        kw["verdict"] = Verdict.DEPLOY_AGENT
        kw["break_even"] = []
        return real.model_construct(*args, **kw)

    monkeypatch.setattr(decision, "VerdictJSON", always_deploys)
    rate = decision._aa_rate(agent, simpler, PREREG, AUDIT, "exact_label", True)
    assert rate == 1.0


def test_flagged_or_other_arms_of_the_wrong_length_are_refused():
    agent, simpler = paired(100, 40)
    with pytest.raises(ValueError):
        run(agent, simpler, flagged=np.zeros(5, dtype=bool))
    with pytest.raises(ValueError):
        run(agent, simpler, others=[paired(1, 1, n=10)[0]])


def test_a_stated_agent_price_is_a_no_op_at_zero_and_adds_to_every_table():
    a = arm("agent", 0.96, prompt=700, completion=26, seed=1)
    s = arm("linear", 0.86, seed=2)
    plain, zero = run(a, s), run(a, s, agent_fixed_usd=0.0)
    assert zero.model_dump() == plain.model_dump()
    priced = run(a, s, agent_fixed_usd=0.01)
    agent_plain = next(x for x in plain.arms if x.name == "agent")
    agent_priced = next(x for x in priced.arms if x.name == "agent")
    for t, c in agent_plain.inference_cost.items():
        assert agent_priced.inference_cost[t] == pytest.approx(c + 0.01)
    linear = [next(x for x in v.arms if x.name == "linear") for v in (plain, priced)]
    assert linear[0].inference_cost == linear[1].inference_cost
    # a dearer agent needs a dearer mistake to break even
    assert all(p.value > q.value for p, q in zip(priced.break_even, plain.break_even, strict=True))


def test_the_aa_check_keeps_a_stated_price_with_the_agent():
    a = arm("agent", 0.96, n=600, prompt=700, completion=26, seed=1)
    s = arm("linear", 0.86, n=600, seed=2)
    assert run(a, s, agent_fixed_usd=0.0).aa_false_winner_rate == run(a, s).aa_false_winner_rate
    # swapped simulations put the agent's run in the other slot: its price goes with it
    swapped = run(s, a, simpler_fixed_usd=0.25, run_aa=False)
    real_agent = next(x for x in swapped.arms if x.name == "agent")
    plain = next(x for x in run(s, a, run_aa=False).arms if x.name == "agent")
    for t, c in plain.inference_cost.items():
        assert real_agent.inference_cost[t] == pytest.approx(c + 0.25)
