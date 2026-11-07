import numpy as np
import pytest
from scipy import stats

from agent_committee import analysis


def test_mcnemar_counts_and_exact_p():
    a = [True] * 30 + [False] * 10 + [True] * 50
    b = [False] * 30 + [True] * 10 + [True] * 50
    only_a, only_b, p = analysis.mcnemar_exact(a, b)
    assert (only_a, only_b) == (30, 10)
    assert p == pytest.approx(stats.binomtest(30, 40, 0.5).pvalue)
    assert p < 0.05


def test_mcnemar_without_discordant_pairs_is_p_one():
    assert analysis.mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)


def test_bootstrap_resamples_cases_together():
    x = np.arange(100, dtype=float)
    boots = analysis.bootstrap_means(np.column_stack([x, 2 * x]), 200, seed=1)
    assert boots.shape == (200, 2)
    assert np.allclose(boots[:, 1], 2 * boots[:, 0])
    assert abs(boots[:, 0].mean() - x.mean()) < 3


def test_bootstrap_is_reproducible_with_a_seed():
    x = np.random.default_rng(0).random((50, 1))
    a = analysis.bootstrap_means(x, 100, seed=7)
    assert np.array_equal(a, analysis.bootstrap_means(x, 100, seed=7))


def test_paired_diff_ci_covers_a_clear_gap():
    a = [True] * 90 + [False] * 10
    b = [True] * 70 + [False] * 30
    diff, lo, hi = analysis.paired_diff_ci(a, b, 2000, 0.95, seed=3)
    assert diff == pytest.approx(0.2)
    assert 0 < lo < diff < hi


def test_paired_diff_ci_straddles_zero_for_equal_arms():
    rng = np.random.default_rng(5)
    a = rng.random(500) > 0.2
    b = rng.random(500) > 0.2
    _, lo, hi = analysis.paired_diff_ci(a, b, 2000, 0.95, seed=3)
    assert lo < 0 < hi


def mcnemar_rule(alpha):
    def rule(x, y, swapped):
        return analysis.mcnemar_exact(x, y)[2] < alpha

    return rule


def test_aa_false_winner_rate_stays_near_alpha():
    rng = np.random.default_rng(11)
    a = rng.random(3000) > 0.15
    b = rng.random(3000) > 0.15
    assert analysis.aa_false_winner_rate(a, b, 400, mcnemar_rule(0.05), seed=2) <= 0.08


def test_aa_ignores_a_real_gap_because_it_swaps_the_arms():
    a = [True] * 950 + [False] * 50
    b = [True] * 800 + [False] * 200
    assert analysis.aa_false_winner_rate(a, b, 300, mcnemar_rule(0.05), seed=4) <= 0.08


def test_aa_with_no_discordant_pairs_has_no_false_winners():
    rate = analysis.aa_false_winner_rate([True, True], [True, True], 100, lambda *_: True, 0)
    assert rate == 0.0


def test_aa_can_fail_when_the_winner_rule_is_miscalibrated():
    # A rule that ignores the data names a winner in every run; the old McNemar-only swap could
    # never exceed alpha, so this is the regression for the A/A being unable to fail.
    rng = np.random.default_rng(3)
    a, b = rng.random(500) > 0.2, rng.random(500) > 0.2
    assert analysis.aa_false_winner_rate(a, b, 50, lambda *_: True, seed=1) == 1.0


def test_paired_arrays_of_unequal_length_are_refused():
    with pytest.raises(ValueError):
        analysis.mcnemar_exact([True], [False, False, True, False])
    with pytest.raises(ValueError):
        analysis.paired_diff_ci([True, False], [True], 100, 0.95, 0)
    with pytest.raises(ValueError):
        analysis.aa_false_winner_rate([True, False], [True], 10, lambda *_: False, 0)
