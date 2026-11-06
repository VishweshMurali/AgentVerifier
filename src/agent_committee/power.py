"""Power analysis for the paired accuracy comparison (exact McNemar, two-sided).

Setup: N paired cases; a fraction `discordance` of them are discordant (exactly one arm right).
With accuracy gap `gap` = P(agent right, other wrong) - P(other right, agent wrong), the discordant
cells are p10 = (d + gap) / 2 and p01 = (d - gap) / 2. Conditional on n_d discordant pairs, the
count of agent-only wins is Binomial(n_d, q) with q = p10 / d; the exact test rejects when that
count is in the two-sided Binomial(n_d, 0.5) tail with probability <= alpha. n_d itself is
Binomial(N, d), so power is computed exactly, with no normal approximation.

`required_n` powers the McNemar test only. The DEPLOY_AGENT rule also needs the bootstrap CI of
the accuracy difference above 0 and the observed difference at least min_gap, so its power is
lower: at a true gap equal to min_gap the observed gap falls short about half the time.
`deploy_rule_power` simulates that full rule with the same functions the decision engine uses.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from agent_committee.analysis import mcnemar_exact, percentile_ci

DEFAULT_DISCORDANCE_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
HEADLINE_DISCORDANCE = 0.15
_STABLE_WINDOW = 50
# fixed so the reported full-rule power is reproducible
DEPLOY_SIM_SEED = 20261105
DEPLOY_SIMS = 1000
DEPLOY_BOOTSTRAP_REPLICATES = 10000  # prereg stats.bootstrap_replicates


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


def deploy_rule_power(
    n: int,
    gap: float,
    discordance: float,
    alpha: float,
    min_gap: float,
    ci_level: float,
    sims: int = DEPLOY_SIMS,
    replicates: int = DEPLOY_BOOTSTRAP_REPLICATES,
    seed: int = DEPLOY_SIM_SEED,
) -> float:
    """Simulated share of runs in which the full DEPLOY_AGENT rule fires at a true `gap`.

    Each run draws n paired cases (agent-only right with probability (d + gap) / 2, simpler-only
    right with (d - gap) / 2, the rest concordant) and applies the decision engine's rule: exact
    McNemar p < alpha, the paired case-bootstrap CI lower bound > 0, and the observed difference
    >= min_gap. See `case_bootstrap_diff` for how the bootstrap is drawn.
    """
    _validate(gap, discordance, alpha)
    if n < 1 or sims < 1 or replicates < 1:
        raise ValueError("n, sims and replicates must be >= 1")
    rng = np.random.default_rng(seed)
    p10, p01 = (discordance + gap) / 2, (discordance - gap) / 2
    fired = 0
    for _ in range(sims):
        u = rng.random(n)
        agent_ok = u >= p01  # the simpler-only cell is the only one the agent gets wrong
        simpler_ok = (u < p01) | (u >= p01 + p10)
        _, _, p = mcnemar_exact(agent_ok, simpler_ok)
        d = agent_ok.astype(float) - simpler_ok.astype(float)
        if p >= alpha or float(np.mean(d)) < min_gap:
            continue
        lo, _ = percentile_ci(case_bootstrap_diff(d, replicates, rng), ci_level)
        fired += lo > 0
    return fired / sims


def case_bootstrap_diff(d: np.ndarray, replicates: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap means of a paired difference d in {-1, 0, 1}, resampling whole cases.

    Resampling n cases with replacement only matters through how many +1 and -1 cases are drawn,
    which is Multinomial(n, shares of +1, -1, 0). Drawing those three counts gives the same
    distribution as `analysis.bootstrap_means` on d, at a fraction of the cost.
    """
    n = len(d)
    shares = np.array([np.sum(d == 1), np.sum(d == -1), np.sum(d == 0)]) / n
    counts = rng.multinomial(n, shares, size=replicates)
    return (counts[:, 0] - counts[:, 1]) / n


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
