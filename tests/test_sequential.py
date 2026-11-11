import numpy as np
import pytest

from agent_committee import sequential as s


@pytest.fixture(scope="module")
def bounds():
    return s.boundaries()


def test_boundaries_shrink_and_end_near_the_fixed_test(bounds):
    assert len(bounds) == s.LOOKS
    assert all(a > b for a, b in zip(bounds, bounds[1:], strict=False))
    assert bounds[0] == pytest.approx(1.959964 / np.sqrt(0.2), abs=1e-3)  # first look: all spent
    assert 2.0 < bounds[-1] < 2.1  # a little above 1.96: the price of four early looks


def test_boundaries_spend_exactly_alpha_by_monte_carlo(bounds):
    t = np.arange(1, s.LOOKS + 1) / s.LOOKS
    rng = np.random.default_rng(7)
    z = np.cumsum(rng.standard_normal((400_000, s.LOOKS)) * np.sqrt(np.diff(np.r_[0, t])), 1)
    crossed = (np.abs(z / np.sqrt(t)) >= bounds).any(axis=1).mean()
    assert crossed == pytest.approx(0.05, abs=0.002)


def test_boundaries_are_deterministic(bounds):
    assert np.array_equal(bounds, s.boundaries())


def test_look_sizes_are_equal_blocks_ending_at_the_total():
    assert s.look_sizes(658) == [132, 263, 395, 526, 658]


def test_no_disagreement_yet_never_stops_either_way(bounds):
    look = s.check(np.zeros(34, dtype=int), 0.10, bounds[0])
    assert look.outcome is None and look.z0 == 0.0


def test_clear_cases_stop_early_in_the_right_direction(bounds):
    agent_better = np.array([1] * 40 + [0] * 160)  # +20 points, no losses
    assert s.run(agent_better, 0.05, bounds).outcome == "agent"
    simpler_better = np.array([-1] * 40 + [0] * 160)
    look = s.run(simpler_better, 0.05, bounds)
    assert look.outcome == "simpler" and look.n < 200


@pytest.mark.parametrize("psi", [0.1, 0.2, 0.3])
def test_wrong_calls_stay_rare_across_disagreement_rates(bounds, psi):
    """The whole rule, simulated (owner decision: false winners about 5% at most)."""
    gap, total = 0.10, 168
    null = s.simulate(total, gap, 0.0, psi, bounds, sims=1500, seed=3)
    real = s.simulate(total, gap, gap, psi, bounds, sims=1500, seed=4)
    assert null["agent"] <= 0.05  # agent declared better with no real gain
    assert real["simpler"] <= 0.05  # a real gain of `gap` ruled out
    assert null["mean_fraction_used"] < 1 and real["mean_fraction_used"] < 1


def test_repeated_interval_contains_the_difference(bounds):
    look = s.check(np.array([1, 0, -1, 1, 0] * 20), 0.05, bounds[-1])
    lo, hi = look.interval
    assert lo < look.diff < hi
