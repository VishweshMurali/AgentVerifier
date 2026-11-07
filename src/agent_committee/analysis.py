"""Paired comparison statistics for the sealed run. Plain code, no LLM.

Exact McNemar on the discordant pairs, a paired bootstrap that resamples whole cases (each case is
its own cluster, so every arm's outcome and cost on a case move together), and an A/A simulation
that measures how often the test names a false winner when the two arms are exchangeable.
"""

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats

CHUNK = 500


def _paired_array(values: Sequence[bool], n: int | None = None) -> np.ndarray:
    """A 1-d bool array; raises if it is not the paired length (no silent broadcasting)."""
    arr = np.asarray(values, dtype=bool)
    if arr.ndim != 1 or (n is not None and len(arr) != n):
        raise ValueError(f"paired arrays must be 1-d and of equal length (got shape {arr.shape})")
    return arr


def mcnemar_exact(a_ok: Sequence[bool], b_ok: Sequence[bool]) -> tuple[int, int, float]:
    """(cases only A got right, cases only B got right, two-sided exact p)."""
    a = _paired_array(a_ok)
    b = _paired_array(b_ok, len(a))
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
    a = _paired_array(a_ok).astype(float)
    b = _paired_array(b_ok, len(a)).astype(float)
    boots = bootstrap_means(np.column_stack([a - b]), replicates, seed)[:, 0]
    lo, hi = percentile_ci(boots, ci_level)
    return float(a.mean() - b.mean()), lo, hi


def aa_false_winner_rate(
    a_ok: Sequence[bool],
    b_ok: Sequence[bool],
    sims: int,
    names_winner: Callable[[np.ndarray, np.ndarray, bool], bool],
    seed: int,
) -> float:
    """Share of simulated A/A runs in which the real winner rule names a winner.

    Under no difference the two arms are exchangeable on every case, so each simulation swaps the
    arms' outcomes case by case with a fair coin (a paired sign flip). `names_winner(x, y, swapped)`
    is the decision engine's own winner rule applied to those outcomes (x in the agent's role, y in
    the simpler arm's; `swapped` is True when the roles are reversed). A winner in either direction
    counts. The rate should sit near or below alpha; a miscalibrated rule shows up here.
    """
    a = _paired_array(a_ok)
    b = _paired_array(b_ok, len(a))
    if not np.any(a != b):
        return 0.0
    rng = np.random.default_rng(seed)
    winners = 0
    for _ in range(sims):
        flip = rng.random(len(a)) < 0.5
        x, y = np.where(flip, b, a), np.where(flip, a, b)
        winners += bool(names_winner(x, y, False) or names_winner(y, x, True))
    return winners / sims
