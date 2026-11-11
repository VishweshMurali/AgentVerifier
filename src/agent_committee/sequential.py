"""Early stopping for quick tests: a group-sequential design with planned looks.

The quick test's sample is split into K equal blocks (looks). After each block, two one-sided
tests are checked against the same Lan-DeMets O'Brien-Fleming boundaries c_1..c_K:

- agent ahead:   z0 = d / se >= c_k        (the agent is better than the simpler model)
- simpler ahead: z1 = (d - gap) / se <= -c_k (a gain of `gap` or more is ruled out)

where d is the mean paired difference (agent right - simpler right) on the cases so far and se
its standard error. The first boundary crossed stops the run. If neither is crossed by the last
look, the answer is NEED_MORE_DATA. The repeated confidence interval d +- c_k * se is valid at
every look, so it replaces the bootstrap interval after a stop.

Boundaries come from numerical integration of the Brownian-motion recursion (Armitage, McPherson
& Rowe 1969; spending function of Lan & DeMets 1983), deterministic and computed here, not
looked up. `simulate` checks the error rates of the whole rule. Owner approved early stopping
for quick tests only (2026-11-11); the sealed run stays fixed-sample.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

LOOKS = 5
GRID = 4001  # grid points per look for the recursion


def obf_spending(t: np.ndarray, alpha: float) -> np.ndarray:
    """Lan-DeMets O'Brien-Fleming-type two-sided alpha spent by information fraction t."""
    z = stats.norm.ppf(1 - alpha / 2)
    return 2 - 2 * stats.norm.cdf(z / np.sqrt(t))


def _crossing(lim: float, step: float, xs: np.ndarray | None, dens: np.ndarray | None) -> float:
    """P(first crossing of +-lim at this look), given the continuation density at the last."""
    sd = np.sqrt(step)
    if xs is None or dens is None:
        return float(1 - (stats.norm.cdf(lim / sd) - stats.norm.cdf(-lim / sd)))
    dx = xs[1] - xs[0]
    # P(stay inside at this look | B(t_prev) = x), integrated against the carried density
    inside = stats.norm.cdf((lim - xs) / sd) - stats.norm.cdf((-lim - xs) / sd)
    return float(np.sum(dens) * dx - np.sum(dens * inside) * dx)


def boundaries(alpha: float = 0.05, looks: int = LOOKS) -> np.ndarray:
    """Two-sided critical z values c_1..c_K for equally spaced looks.

    One-sided tests at each side use alpha/2 per side, so each of the two directional tests has
    a false-stop rate of at most alpha/2 under its null.
    """
    t = np.arange(1, looks + 1) / looks
    spend = obf_spending(t, alpha)
    out: list[float] = []
    # density of B(t) on the continuation region, on a grid in the B scale
    xs: np.ndarray | None = None
    dens: np.ndarray | None = None
    prev_t = 0.0
    spent = 0.0
    for k in range(looks):
        step = t[k] - prev_t
        target = spend[k] - spent
        c = optimize.brentq(
            lambda c, k=k, step=step, xs=xs, dens=dens, target=target: (
                _crossing(c * np.sqrt(t[k]), step, xs, dens) - target
            ),
            0.5,
            12.0,
            xtol=1e-10,
        )
        out.append(c)
        spent += target
        lim = c * np.sqrt(t[k])
        new_xs = np.linspace(-lim, lim, GRID)
        if xs is None:
            new_dens = stats.norm.pdf(new_xs / np.sqrt(step)) / np.sqrt(step)
        else:
            dx = xs[1] - xs[0]
            kern = stats.norm.pdf((new_xs[:, None] - xs[None, :]) / np.sqrt(step)) / np.sqrt(step)
            new_dens = kern @ dens * dx
        xs, dens, prev_t = new_xs, new_dens, t[k]
    return np.array(out)


@dataclass(frozen=True)
class Look:
    n: int
    diff: float
    se0: float  # standard error behind the "agent ahead" test (lower side)
    se1: float  # floored standard error behind the "gain ruled out" test (upper side)
    z0: float
    z1: float
    bound: float
    outcome: str | None  # "agent", "simpler" or None (continue)

    @property
    def interval(self) -> tuple[float, float]:
        """Repeated confidence interval, valid at every look. Each side matches its own test:
        lower > 0 exactly when "agent ahead" is crossed, upper < gap when "simpler ahead" is."""
        return self.diff - self.bound * self.se0, self.diff + self.bound * self.se1


def look_sizes(total: int, looks: int = LOOKS) -> list[int]:
    """Cumulative sample sizes at each look: equal blocks, the last one at `total`."""
    return [round(total * (k + 1) / looks) for k in range(looks)]


def check(d: np.ndarray, gap: float, bound: float) -> Look:
    """One look on the paired differences so far (d in {-1, 0, 1})."""
    n = len(d)
    diff = float(np.mean(d))
    var = float(np.mean(d * d) - diff * diff)
    # z0 tests "no gain": with no disagreements yet there is no evidence either way
    z0 = diff / np.sqrt(var / n) if var > 0 else 0.0
    # z1 tests "gain >= gap": a true gain of `gap` needs at least that much disagreement, so its
    # variance is at least gap * (1 - gap); without this floor, a few early cases with no
    # disagreement would rule the gain out falsely (seen in simulation: 5.4% at 168 cases)
    se1 = np.sqrt(max(var, gap * (1 - gap)) / n)
    z1 = (diff - gap) / se1
    outcome = "agent" if z0 >= bound else "simpler" if z1 <= -bound else None
    se0 = float(np.sqrt(var / n)) if var > 0 else float(se1)
    return Look(n, diff, se0, float(se1), float(z0), float(z1), float(bound), outcome)


def run(d: np.ndarray, gap: float, bounds: np.ndarray) -> Look:
    """Apply the rule to the paired differences in their planned order; returns the stopping
    look (or the last look, outcome None, when no boundary is crossed)."""
    sizes = look_sizes(len(d), len(bounds))
    for n, c in zip(sizes, bounds, strict=True):
        last = check(d[:n], gap, c)
        if last.outcome is not None:
            return last
    return last


def simulate(
    total: int,
    gap: float,
    true_diff: float,
    disagreement: float,
    bounds: np.ndarray,
    sims: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Share of runs ending 'agent', 'simpler' or undecided, and the mean fraction of `total`
    used, when the agent-only and simpler-only right rates are (psi +- true_diff) / 2."""
    rng = np.random.default_rng(seed)
    p_up, p_down = (disagreement + true_diff) / 2, (disagreement - true_diff) / 2
    if p_down < 0 or p_up > 1:
        raise ValueError("true_diff must be within the disagreement rate")
    counts = {"agent": 0, "simpler": 0, None: 0}
    used = 0.0
    for _ in range(sims):
        u = rng.random(total)
        d = np.where(u < p_up, 1, np.where(u < p_up + p_down, -1, 0))
        look = run(d, gap, bounds)
        counts[look.outcome] += 1
        used += look.n / total
    return {
        "agent": counts["agent"] / sims,
        "simpler": counts["simpler"] / sims,
        "undecided": counts[None] / sims,
        "mean_fraction_used": used / sims,
    }
