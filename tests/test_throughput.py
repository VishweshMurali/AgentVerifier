import pytest

from agent_committee.throughput import calls_per_hour


def test_calls_per_hour_arithmetic():
    assert calls_per_hour(30, 60) == pytest.approx(1800)
    assert calls_per_hour(0, 10) == 0


@pytest.mark.parametrize("n,t", [(-1, 1), (5, 0), (5, -2)])
def test_calls_per_hour_rejects_invalid_input(n, t):
    with pytest.raises(ValueError):
        calls_per_hour(n, t)
