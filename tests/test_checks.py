import importlib.util
from pathlib import Path

import numpy as np
import pytest

from agent_committee import checks
from agent_committee.prereg import load_prereg
from agent_committee.schemas import (
    AuditBundle,
    Case,
    CheckEffect,
    CheckResult,
    ChecksConfig,
    EvidenceLevel,
    Verdict,
    VerdictJSON,
)

CFG = load_prereg().checks
RNG = np.random.default_rng(0)
WORDS = "card transfer money account balance fee refund top up pin cash atm limit app".split()


def sentences(n, words=WORDS, seed=0):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(words, size=6)) for _ in range(n)]


def test_prereg_sets_every_check_limit():
    assert isinstance(CFG, ChecksConfig)
    assert CFG.null_rate_band == (0.07, 0.13)


def test_band_must_be_ordered():
    with pytest.raises(ValueError):
        ChecksConfig(**{**CFG.model_dump(), "null_rate_band": (0.2, 0.1)})


def test_newcombe_matches_the_published_example():
    # Newcombe (1998), example (a): 56/70 vs 48/80 -> 0.2000, CI 0.0524 to 0.3339
    d, lo, hi = checks.newcombe_diff(56, 70, 48, 80)
    assert (round(d, 4), round(lo, 4), round(hi, 4)) == (0.2, 0.0524, 0.3339)


def test_tune_gap_passes_a_steady_arm_and_fails_a_collapse():
    assert checks.tune_gap("linear", 440, 488, 1325, 1472, CFG).passed
    fail = checks.tune_gap("agent", 440, 488, 1100, 1472, CFG)
    assert not fail.passed and fail.effect == CheckEffect.NEED_MORE_DATA


def test_null_rate_is_report_only():
    assert checks.null_rate(130, 1472, CFG).passed
    r = checks.null_rate(300, 1472, CFG)
    assert not r.passed and r.effect == CheckEffect.REPORT


def test_copies_are_found_exactly_and_nearly_and_dropped():
    reference = [Case(id="r1", text="I am still waiting for my new card to arrive.", label="a")]
    test = [
        Case(id="t1", text="i am STILL waiting for my new card to arrive!", label="a"),
        Case(id="t2", text="I am still waiting for my new card to arrive..", label="a"),
        Case(id="t3", text="How do I change my PIN at an ATM?", label="b"),
    ]
    result, kept = checks.leakage(test, reference, CFG)
    assert [c.id for c in kept] == ["t3"]
    assert not result.passed and result.effect == CheckEffect.REPORT


def test_near_copy_threshold_is_respected():
    a = "please tell me why my top up via bank transfer has not arrived yet"
    b = "please tell me why my card payment via apple pay was declined today"
    assert not checks.find_copies([b], [a], CFG.leak_jaccard)[0]


def test_power_cap():
    assert checks.power_cap(3267, CFG).passed
    assert not checks.power_cap(3000, CFG).passed


def test_small_samples_do_not_raise_drift_on_noise():
    a, b = np.random.default_rng(1).random(40), np.random.default_rng(2).random(40)
    assert checks.score_drift(a, b, CFG, own_split=True).passed


def test_score_drift_effect_depends_on_whose_data():
    same = checks.score_drift(RNG.random(2000), RNG.random(2000), CFG, own_split=True)
    assert same.passed
    shifted = RNG.random(2000) * 0.5
    assert checks.score_drift(RNG.random(2000), shifted, CFG, True).effect == CheckEffect.INVALID
    new = checks.score_drift(RNG.random(2000), shifted, CFG, own_split=False)
    assert not new.passed and new.effect == CheckEffect.NEED_MORE_DATA


def test_domain_auc_sees_new_vocabulary_but_not_a_random_split():
    texts = sentences(400)
    assert checks.domain_auc(texts[:200], texts[200:], CFG, own_split=False).passed
    other = sentences(200, "mortgage loan interest rate house deposit broker".split(), seed=1)
    assert not checks.domain_auc(texts[:200], other, CFG, own_split=False).passed


def test_label_mix():
    assert checks.label_mix(["a", "b"] * 50, ["a", "b"] * 50, CFG).passed
    assert checks.total_variation(["a"] * 10, ["b"] * 10) == 1.0


def test_benjamini_hochberg():
    # step-up limits 0.0125, 0.025, 0.0375, 0.05
    assert checks.benjamini_hochberg([0.001, 0.02, 0.04, 0.9], 0.05) == [0, 1]
    assert checks.benjamini_hochberg([0.001, 0.02, 0.03, 0.9], 0.05) == [0, 1, 2]
    assert checks.benjamini_hochberg([0.5, 0.9], 0.05) == []


def test_per_label_bias_finds_one_bad_label():
    cfg = ChecksConfig(**{**CFG.model_dump(), "permutations": 500})
    labels = [f"l{i % 10}" for i in range(1000)]
    rng = np.random.default_rng(3)
    a, b = rng.random(1000) > 0.1, rng.random(1000) > 0.1
    fair = checks.per_label_bias(a, b, labels, cfg, alpha=0.05)
    assert fair.passed and "none" in fair.detail
    agent = [ok and lab != "l3" for ok, lab in zip(a, labels, strict=True)]
    bad = checks.per_label_bias(agent, b, labels, cfg, alpha=0.05)
    assert not bad.passed and "['l3']" in bad.detail and bad.effect == CheckEffect.REPORT


def test_per_label_bias_ignores_a_gap_shared_by_every_label():
    cfg = ChecksConfig(**{**CFG.model_dump(), "permutations": 500})
    labels = [f"l{i // 200}" for i in range(2000)]  # each label holds two full cycles of 100
    simpler = [i % 100 >= 10 for i in range(2000)]  # 90% right on every label
    agent = [i % 100 >= 6 for i in range(2000)]  # 94% right: the same +4 points on every label
    flat = checks.per_label_bias(agent, simpler, labels, cfg, alpha=0.05)
    assert flat.passed and "none" in flat.detail


def test_flagged_split():
    flagged = [i % 10 == 0 for i in range(1000)]
    even = checks.flagged_split([True] * 1000, [True] * 1000, flagged, 500, 0.95)
    assert even.passed and even.value == 0
    agent = [not f for f in flagged]
    lopsided = checks.flagged_split(agent, [True] * 1000, flagged, 500, 0.95)
    assert not lopsided.passed and lopsided.value == -1


def test_false_alarm_rate_on_one_dataset_is_low():
    r = checks.false_alarm_rate(sentences(300), RNG.random(300), CFG, reps=5)
    assert r.passed and r.effect == CheckEffect.INVALID


AUDIT = AuditBundle(
    git_commit="c",
    prereg_sha256="p",
    dev_report_sha256="d",
    test_sha256="t",
    model_digest="m",
    layer_fingerprint="f",
    predictions_sha256="x",
)


def verdict(v, *failed):
    return VerdictJSON(
        verdict=v,
        evidence_level=EvidenceLevel.REAL,
        audit=AUDIT,
        reasons=["r"],
        scorer="exact_label",
        quality_floor_passed=True,
        accuracy={"linear": 0.9},
        checks=[CheckResult(name="c", value=1.0, passed=False, effect=e) for e in failed],
    )


def test_failed_invalid_check_forces_invalid():
    with pytest.raises(ValueError):
        verdict(Verdict.NEED_MORE_DATA, CheckEffect.INVALID)
    assert verdict(Verdict.INVALID, CheckEffect.INVALID).verdict == Verdict.INVALID


def test_failed_need_more_data_check_blocks_a_winner():
    with pytest.raises(ValueError):
        verdict(Verdict.USE_SIMPLER, CheckEffect.NEED_MORE_DATA)
    assert verdict(Verdict.NEED_MORE_DATA, CheckEffect.NEED_MORE_DATA)


def test_report_checks_never_change_the_verdict():
    assert verdict(Verdict.USE_SIMPLER, CheckEffect.REPORT).verdict == Verdict.USE_SIMPLER


def test_validate_checks_script_passes_on_dev(monkeypatch, capsys):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    path = root / "scripts/validate_checks.py"
    spec = importlib.util.spec_from_file_location("validate_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--reps", "1"]) == 0
    out = capsys.readouterr().out
    for name in ("tune_gap[linear]", "score_drift", "domain_auc", "false_alarm_rate"):
        assert "PASS" in next(line for line in out.splitlines() if name in line)
    assert "1/1 random halves" not in out
