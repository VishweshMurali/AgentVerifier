import pytest

from agent_committee.power import mcnemar_power, power_table, required_n


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


def test_table_skips_discordance_below_gap():
    assert [r.discordance for r in power_table(0.10, grid=(0.05, 0.15))] == [0.15]


def test_validation():
    with pytest.raises(ValueError):
        mcnemar_power(100, 0.2, 0.1)
