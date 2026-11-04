"""Paired comparison statistics for the sealed run. Plain code, no LLM.

Exact McNemar on the discordant pairs, a paired bootstrap that resamples whole cases (each case is
its own cluster, so every arm's outcome and cost on a case move together), and an A/A simulation
that measures how often the test names a false winner when the two arms are exchangeable.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats

CHUNK = 500


def mcnemar_exact(a_ok: Sequence[bool], b_ok: Sequence[bool]) -> tuple[int, int, float]:
    """(cases only A got right, cases only B got right, two-sided exact p)."""
    a = np.asarray(a_ok, dtype=bool)
    b = np.asarray(b_ok, dtype=bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(only_a, n, 0.5).pvalue)
    return only_a, only_b, p


def bootstrap_means(columns: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    """Column means over `replicates` case resamples; one row per replicate.

    `columns` is cases x quantities. Resampling uses multinomial case counts, so every quantity in
    a replicate is computed on the same resampled cases.
    """
    data = np.asarray(columns, dtype=float)
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((replicates, data.shape[1]))
    weights = np.full(n, 1.0 / n)
    for start in range(0, replicates, CHUNK):
        m = min(CHUNK, replicates - start)
        counts = rng.multinomial(n, weights, size=m)
        out[start : start + m] = counts @ data / n
    return out


def percentile_ci(samples: np.ndarray, ci_level: float) -> tuple[float, float]:
    tail = (1 - ci_level) / 2
    lo, hi = np.quantile(np.asarray(samples, dtype=float), [tail, 1 - tail])
    return float(lo), float(hi)


def paired_diff_ci(
    a_ok: Sequence[bool], b_ok: Sequence[bool], replicates: int, ci_level: float, seed: int
) -> tuple[float, float, float]:
    """Accuracy of A minus accuracy of B, with a paired case-bootstrap percentile CI."""
    a = np.asarray(a_ok, dtype=float)
    b = np.asarray(b_ok, dtype=float)
    boots = bootstrap_means(np.column_stack([a - b]), replicates, seed)[:, 0]
    lo, hi = percentile_ci(boots, ci_level)
    return float(a.mean() - b.mean()), lo, hi


def aa_false_winner_rate(
    a_ok: Sequence[bool],
    b_ok: Sequence[bool],
    sims: int,
    alpha: float,
    min_gap: float,
    ci_level: float,
    replicates: int,
    seed: int,
) -> float:
    """Share of simulated A/A runs in which the full winner rule names a winner.

    Under no difference the two arms are exchangeable on every case, so each simulation swaps the
    arms' outcomes case by case with a fair coin (a paired sign flip) and applies the same rule
    the decision engine uses for a winner in either direction: exact McNemar p < alpha, the
    bootstrap CI clear of 0, and a gap of at least min_gap. The rate should sit near or below alpha.
    """
    a = np.asarray(a_ok, dtype=bool)
    b = np.asarray(b_ok, dtype=bool)
    n = len(a)
    discordant = int(np.sum(a != b))
    if discordant == 0:
        return 0.0
    p_by_k = np.array([stats.binomtest(k, discordant, 0.5).pvalue for k in range(discordant + 1)])
    rng = np.random.default_rng(seed)
    weights = np.full(n, 1.0 / n)
    tail = (1 - ci_level) / 2
    winners = 0
    for _ in range(sims):
        flip = rng.random(n) < 0.5
        d = np.where(flip, b, a).astype(float) - np.where(flip, a, b).astype(float)
        p = p_by_k[int(np.sum(d == 1))]
        if p >= alpha or abs(d.mean()) < min_gap:
            continue
        boots = rng.multinomial(n, weights, size=replicates) @ d / n
        lo, hi = np.quantile(boots, [tail, 1 - tail])
        winners += bool(lo > 0 or hi < 0)
    return winners / sims
