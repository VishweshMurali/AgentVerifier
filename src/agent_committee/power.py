"""Power analysis for the paired accuracy comparison (exact McNemar, two-sided).

Setup: N paired cases; a fraction `discordance` of them are discordant (exactly one arm right).
With accuracy gap `gap` = P(agent right, other wrong) - P(other right, agent wrong), the discordant
cells are p10 = (d + gap) / 2 and p01 = (d - gap) / 2. Conditional on n_d discordant pairs, the
count of agent-only wins is Binomial(n_d, q) with q = p10 / d; the exact test rejects when that
count is in the two-sided Binomial(n_d, 0.5) tail with probability <= alpha. n_d itself is
Binomial(N, d), so power is computed exactly, with no normal approximation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

DEFAULT_DISCORDANCE_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
HEADLINE_DISCORDANCE = 0.15
_STABLE_WINDOW = 50


def _validate(gap: float, discordance: float, alpha: float) -> None:
    if not 0 < gap < 1:
        raise ValueError("gap must be in (0, 1)")
    if not gap <= discordance <= 1:
        raise ValueError("discordance must be in [gap, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def critical_values(max_discordant: int, alpha: float) -> np.ndarray:
    """Smallest k per n_d (index n_d) so that 2 * P(Bin(n_d, .5) >= k) <= alpha. n_d+1 = never."""
    nd = np.arange(max_discordant + 1)
    # isf(a/2) is the smallest j with P(X > j) <= a/2, so k = j + 1 has 2 * P(X >= k) <= a.
    crit = binom.isf(alpha / 2, nd, 0.5).astype(int) + 1
    return np.minimum(crit, nd + 1)


def mcnemar_power(n: int, gap: float, discordance: float, alpha: float = 0.05) -> float:
    """Exact power of the two-sided exact McNemar test (rejecting in either direction)."""
    _validate(gap, discordance, alpha)
    if n < 1:
        raise ValueError("n must be >= 1")
    q = (discordance + gap) / (2 * discordance)
    nd = np.arange(n + 1)
    # restrict to the mass of n_d that matters
    mean, sd = n * discordance, (n * discordance * (1 - discordance)) ** 0.5
    lo, hi = max(0, int(mean - 8 * sd)), min(n, int(mean + 8 * sd) + 1)
    nd = nd[lo : hi + 1]
    crit = critical_values(hi, alpha)[nd]
    # reject if wins >= crit, or wins <= nd - crit (symmetric lower tail); the two are disjoint
    # because crit > nd / 2 whenever a rejection is possible at all.
    can = crit <= nd
    p_upper = binom.sf(crit - 1, nd, q)
    p_lower = binom.cdf(nd - crit, nd, q)
    p_reject = np.where(can, p_upper + p_lower, 0.0)
    return float(np.sum(binom.pmf(nd, n, discordance) * p_reject))


def required_n(
    gap: float, discordance: float, alpha: float = 0.05, power: float = 0.8, n_max: int = 200_000
) -> int:
    """Smallest N whose power stays >= target for the next _STABLE_WINDOW sample sizes.

    Exact-test power is sawtooth in N, so the first crossing is not trusted on its own.
    """
    _validate(gap, discordance, alpha)
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    # Bisect on the (nearly monotone) power curve, back off past the sawtooth, then scan up.
    lo, hi = 1, min(1000, n_max)
    while mcnemar_power(hi, gap, discordance, alpha) < power:
        if hi >= n_max:
            raise ValueError(f"required N exceeds n_max={n_max}")
        lo, hi = hi, min(hi * 2, n_max)
    while lo < hi:
        mid = (lo + hi) // 2
        if mcnemar_power(mid, gap, discordance, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    n = max(1, lo - 4 * _STABLE_WINDOW)
    run = 0
    first = None
    while n <= n_max:
        if mcnemar_power(n, gap, discordance, alpha) >= power:
            first = n if run == 0 else first
            run += 1
            # The window is cut short at n_max: the search never looks past the caller's bound.
            if run >= _STABLE_WINDOW or n == n_max:
                return first  # type: ignore[return-value]
        else:
            run, first = 0, None
        n += 1
    raise ValueError(f"required N exceeds n_max={n_max}")


@dataclass(frozen=True)
class PowerRow:
    discordance: float
    required_n: int


def power_table(
    gap: float,
    alpha: float = 0.05,
    power: float = 0.8,
    grid: tuple[float, ...] = DEFAULT_DISCORDANCE_GRID,
) -> list[PowerRow]:
    return [PowerRow(d, required_n(gap, d, alpha, power)) for d in grid if d >= gap]
