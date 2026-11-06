import importlib.util
from pathlib import Path

import numpy as np
import pytest

from agent_committee.analysis import bootstrap_means
from agent_committee.power import (
    HEADLINE_DISCORDANCE,
    case_bootstrap_diff,
    deploy_rule_power,
    mcnemar_power,
    power_table,
    required_n,
)
from agent_committee.prereg import load_prereg

ROOT = Path(__file__).resolve().parents[1]


def test_power_increases_with_n():
    assert mcnemar_power(500, 0.02, 0.15) < mcnemar_power(5000, 0.02, 0.15)


def test_rejection_rate_near_null_does_not_exceed_alpha():
    assert mcnemar_power(2000, 1e-9, 0.15) <= 0.05 + 1e-6


def test_required_n_meets_target():
    n = required_n(0.02, 0.15)
    assert mcnemar_power(n, 0.02, 0.15) >= 0.8
    assert mcnemar_power(n - 1, 0.02, 0.15) < 0.8 or mcnemar_power(n - 40, 0.02, 0.15) < 0.8


def test_required_n_monotone_in_discordance_and_gap():
    assert required_n(0.02, 0.10) < required_n(0.02, 0.20)
    assert required_n(0.04, 0.15) < required_n(0.02, 0.15)


def test_close_to_normal_approximation():
    n = required_n(0.02, 0.15)
    approx = (1.96 + 0.8416) ** 2 * 0.15 / 0.02**2
    assert approx * 0.9 < n < approx * 1.2


def test_required_n_respects_but_does_not_overshoot_n_max():
    assert required_n(0.02, 0.15, n_max=3271) == required_n(0.02, 0.15)
    with pytest.raises(ValueError, match="exceeds n_max"):
        required_n(0.02, 0.15, n_max=3000)


def test_table_skips_discordance_below_gap():
    assert [r.discordance for r in power_table(0.10, grid=(0.05, 0.15))] == [0.15]


def test_validation():
    with pytest.raises(ValueError):
        mcnemar_power(100, 0.2, 0.1)


def test_prereg_min_test_n_is_the_headline_required_n():
    prereg = load_prereg()
    expected = required_n(prereg.min_gap, HEADLINE_DISCORDANCE, prereg.alpha, prereg.power)
    assert prereg.checks.min_test_n == expected


def test_case_bootstrap_diff_matches_the_decision_engines_bootstrap():
    d = np.r_[np.ones(300), -np.ones(200), np.zeros(1500)]
    fast = case_bootstrap_diff(d, 4000, np.random.default_rng(1))
    slow = bootstrap_means(d[:, None], 4000, seed=2)[:, 0]
    assert fast.mean() == pytest.approx(d.mean(), abs=0.002)
    assert fast.std() == pytest.approx(slow.std(), rel=0.05)
    assert np.quantile(fast, [0.025, 0.975]) == pytest.approx(
        np.quantile(slow, [0.025, 0.975]), abs=0.003
    )


def test_deploy_rule_power_is_reproducible_and_below_mcnemar_power():
    args = (3028, 0.02, 0.15, 0.05, 0.02, 0.95)
    first = deploy_rule_power(*args, sims=300, replicates=2000)
    assert first == deploy_rule_power(*args, sims=300, replicates=2000)
    assert 0.35 < first < 0.7 < mcnemar_power(3028, 0.02, 0.15)


def test_deploy_rule_power_nears_one_for_a_gap_well_above_min_gap():
    assert deploy_rule_power(3028, 0.05, 0.15, 0.05, 0.02, 0.95, sims=100, replicates=1000) > 0.95


def test_deploy_rule_power_validates_inputs():
    with pytest.raises(ValueError):
        deploy_rule_power(0, 0.02, 0.15, 0.05, 0.02, 0.95)
    with pytest.raises(ValueError):
        deploy_rule_power(100, 0.2, 0.1, 0.05, 0.02, 0.95)


def test_power_analysis_script_reports_the_full_rule_power(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    spec = importlib.util.spec_from_file_location(
        "power_analysis", ROOT / "scripts/power_analysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(["--sims", "50"])
    out = capsys.readouterr().out
    assert "3,028" in out and "<- headline" in out
    assert "full-rule power" in out
    assert "LOWER" in out and "LOWER BOUND" not in out
