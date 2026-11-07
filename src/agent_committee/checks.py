"""Pre-registered checks for overfitting, leakage, drift and bias. Plain code, no LLM.

Limits come from prereg.yaml `checks`. Each check returns a CheckResult whose `effect` says what a
failure does to the verdict, and VerdictJSON enforces it; REPORT checks never change the verdict.
Drift on our own sealed split means the split code is broken (INVALID); drift on new scenario data
means the Banking77 result does not carry over (NEED_MORE_DATA). The drift checks need no labels.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline, make_union

from agent_committee.schemas import Case, CheckEffect, CheckResult, ChecksConfig

Z95 = float(stats.norm.ppf(0.975))
NGRAM = 5
CHUNK = 256


def _wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """p1 - p2 and its 95% CI for two independent proportions (Newcombe hybrid Wilson)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = _wilson(k1, n1)
    l2, u2 = _wilson(k2, n2)
    d = p1 - p2
    return d, d - math.hypot(p1 - l1, u2 - p2), d + math.hypot(u1 - p1, p2 - l2)


def tune_gap(
    arm: str,
    tune_correct: int,
    tune_n: int,
    held_correct: int,
    held_n: int,
    cfg: ChecksConfig,
    reference: str = "tune",
) -> CheckResult:
    """Overfitting to settings picked on `reference` data: held-out accuracy may not fall clearly
    below it. Phase D passes the DEV eval scores as the reference and the sealed test as held-out.
    """
    d, lo, hi = newcombe_diff(held_correct, held_n, tune_correct, tune_n)
    return CheckResult(
        name=f"{reference}_gap[{arm}]",
        value=lo,
        low=-cfg.max_tune_drop,
        passed=lo > -cfg.max_tune_drop,
        effect=CheckEffect.NEED_MORE_DATA,
        detail=f"held-out minus {reference} {d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]",
    )


def null_rate(flagged: int, n: int, cfg: ChecksConfig, expected: float = 0.10) -> CheckResult:
    lo, hi = cfg.null_rate_band
    rate = flagged / n
    p = stats.binomtest(flagged, n, expected).pvalue
    return CheckResult(
        name="null_rate",
        value=rate,
        low=lo,
        high=hi,
        passed=lo <= rate <= hi,
        effect=CheckEffect.REPORT,
        detail=f"{flagged}/{n} flagged, binomial p={p:.3f} against {expected}",
    )


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_copies(texts: Sequence[str], reference: Sequence[str], jaccard: float) -> np.ndarray:
    """True where a text repeats a reference text exactly or by char 5-gram Jaccard >= `jaccard`."""
    norm_ref = [normalize(t) for t in reference]
    norm = [normalize(t) for t in texts]
    exact = set(norm_ref)
    copied = np.array([t in exact for t in norm], dtype=bool)
    vec = CountVectorizer(analyzer="char", ngram_range=(NGRAM, NGRAM), binary=True, lowercase=False)
    # Fit on both sides: n-grams only in the text still count in its size, so Jaccard is exact.
    vec.fit(norm_ref + norm)
    ref = vec.transform(norm_ref)
    new = vec.transform(norm)
    ref_size = np.asarray(ref.sum(axis=1)).ravel()
    new_size = np.asarray(new.sum(axis=1)).ravel()
    for start in range(0, new.shape[0], CHUNK):
        rows = slice(start, start + CHUNK)
        inter = (new[rows] @ ref.T).toarray()
        union = new_size[rows, None] + ref_size[None, :] - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)
        copied[rows] |= jac.max(axis=1, initial=0.0) >= jaccard
    return copied


def leakage(
    test: Sequence[Case], reference: Sequence[Case], cfg: ChecksConfig
) -> tuple[CheckResult, list[Case]]:
    """Test texts copied from DEV. Copies are always dropped from the primary result, so the
    rate is reported only; `power_cap` guards how many cases are left."""
    copied = find_copies([c.text for c in test], [c.text for c in reference], cfg.leak_jaccard)
    rate = float(copied.mean()) if len(test) else 0.0
    kept = [c for c, dup in zip(test, copied, strict=True) if not dup]
    result = CheckResult(
        name="leakage",
        value=rate,
        high=cfg.max_leak_rate,
        passed=rate <= cfg.max_leak_rate,
        effect=CheckEffect.REPORT,
        detail=f"{int(copied.sum())}/{len(test)} copied from DEV; {len(kept)} kept",
    )
    return result, kept


def power_cap(n_kept: int, cfg: ChecksConfig) -> CheckResult:
    return CheckResult(
        name="power_cap",
        value=float(n_kept),
        low=float(cfg.min_test_n),
        passed=n_kept >= cfg.min_test_n,
        effect=CheckEffect.NEED_MORE_DATA,
        detail=f"{n_kept} test cases after dropping copies; power needs {cfg.min_test_n}",
    )


def _drift_effect(own_split: bool) -> CheckEffect:
    return CheckEffect.INVALID if own_split else CheckEffect.NEED_MORE_DATA


def score_drift(
    ref_scores: Sequence[float], new_scores: Sequence[float], cfg: ChecksConfig, own_split: bool
) -> CheckResult:
    """Has the linear layer's confidence shifted? Two-sample KS: D over the limit and p < alpha.

    D alone fires on noise in small samples; p alone fires on tiny shifts in large ones.
    `ref_scores` must be held-out scores: the layer is more confident on its own training pool.
    """
    res = stats.ks_2samp(ref_scores, new_scores)
    drifted = res.statistic > cfg.max_score_ks and res.pvalue < cfg.drift_alpha
    return CheckResult(
        name="score_drift",
        value=float(res.statistic),
        high=cfg.max_score_ks,
        passed=not drifted,
        effect=_drift_effect(own_split),
        detail=f"KS D={res.statistic:.4f}, p={res.pvalue:.3g}",
    )


def domain_auc(
    ref_texts: Sequence[str],
    new_texts: Sequence[str],
    cfg: ChecksConfig,
    own_split: bool,
    seed: int = 0,
) -> CheckResult:
    """Can a classifier tell DEV from the new texts? 5-fold AUC; 0.5 means indistinguishable.

    Drift needs AUC over the limit and a Mann-Whitney p (AUC's own test) below alpha.
    """
    texts = [*ref_texts, *new_texts]
    y = np.r_[np.zeros(len(ref_texts)), np.ones(len(new_texts))]
    model = make_pipeline(
        make_union(
            TfidfVectorizer(sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
        ),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, texts, y, cv=folds, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    p = float(stats.mannwhitneyu(proba[y == 1], proba[y == 0], alternative="greater").pvalue)
    return CheckResult(
        name="domain_auc",
        value=auc,
        high=cfg.max_domain_auc,
        passed=not (auc > cfg.max_domain_auc and p < cfg.drift_alpha),
        effect=_drift_effect(own_split),
        detail=f"5-fold AUC={auc:.4f}, p={p:.3g}, {len(ref_texts)} DEV vs {len(new_texts)} new",
    )


def total_variation(a: Sequence[str], b: Sequence[str]) -> float:
    pa, pb = Counter(a), Counter(b)
    return 0.5 * sum(abs(pa[k] / len(a) - pb[k] / len(b)) for k in pa.keys() | pb.keys())


def label_mix(
    ref_labels: Sequence[str], new_labels: Sequence[str], cfg: ChecksConfig
) -> CheckResult:
    """Label shares, DEV vs new. On unlabelled data pass the linear layer's predicted labels."""
    tv = total_variation(ref_labels, new_labels)
    return CheckResult(
        name="label_mix",
        value=tv,
        high=cfg.max_label_tv,
        passed=tv <= cfg.max_label_tv,
        effect=CheckEffect.REPORT,
        detail=f"total variation {tv:.4f}",
    )


def benjamini_hochberg(pvalues: Sequence[float], alpha: float) -> list[int]:
    """Indices rejected at false-discovery rate `alpha`."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    passing = [i for i, j in enumerate(order, 1) if pvalues[j] <= i / m * alpha]
    return sorted(int(j) for j in order[: max(passing, default=0)])


def per_label_bias(
    agent_ok: Sequence[bool],
    simpler_ok: Sequence[bool],
    labels: Sequence[str],
    cfg: ChecksConfig,
    alpha: float,
    seed: int = 0,
) -> CheckResult:
    """Does the agent-minus-simpler gap differ by label? Label-shuffle omnibus test, then BH.

    The statistic is the largest distance of a label's gap from the overall gap, so a gap shared
    evenly by every label does not count. Per label, the agent's share of discordant wins is
    tested against its overall share. Report only: about 40 test cases per label gives power for
    large differences only.
    """
    d = np.asarray(agent_ok, dtype=float) - np.asarray(simpler_ok, dtype=float)
    names, idx = np.unique(np.asarray(labels), return_inverse=True)
    counts = np.bincount(idx, minlength=len(names))

    def spread(groups: np.ndarray) -> float:
        means = np.bincount(groups, weights=d, minlength=len(names)) / counts
        return float(np.max(np.abs(means - d.mean())))

    observed = spread(idx)
    rng = np.random.default_rng(seed)
    hits = sum(spread(rng.permutation(idx)) >= observed - 1e-12 for _ in range(cfg.permutations))
    p = (1 + hits) / (1 + cfg.permutations)
    win_share = np.sum(d == 1) / max(int(np.sum(d != 0)), 1)
    per_label = []
    for j in range(len(names)):
        wins = int(np.sum(d[idx == j] == 1))
        losses = int(np.sum(d[idx == j] == -1))
        tested = 0 < win_share < 1 and wins + losses
        per_label.append(stats.binomtest(wins, wins + losses, win_share).pvalue if tested else 1.0)
    flagged = [str(names[j]) for j in benjamini_hochberg(per_label, alpha)]
    return CheckResult(
        name="per_label_bias",
        value=p,
        low=alpha,
        passed=p >= alpha,
        effect=CheckEffect.REPORT,
        detail=(
            f"overall gap {d.mean():+.3f}; largest label departure {observed:.3f}; "
            f"BH-significant labels: {flagged or 'none'}"
        ),
    )


def flagged_split(
    agent_ok: Sequence[bool],
    simpler_ok: Sequence[bool],
    flagged: Sequence[bool],
    replicates: int,
    ci_level: float,
    seed: int = 0,
) -> CheckResult:
    """Agent-minus-simpler gap on Null-flagged cases minus the gap on the rest, case bootstrap."""
    d = np.asarray(agent_ok, dtype=float) - np.asarray(simpler_ok, dtype=float)
    f = np.asarray(flagged, dtype=bool)
    if f.all() or not f.any():
        return CheckResult(
            name="flagged_split",
            value=0.0,
            passed=True,
            effect=CheckEffect.REPORT,
            detail="all or no cases flagged; nothing to compare",
        )
    point = d[f].mean() - d[~f].mean()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(replicates):
        i = rng.integers(0, len(d), len(d))
        fi = f[i]
        if fi.any() and not fi.all():
            boots.append(d[i][fi].mean() - d[i][~fi].mean())
    tail = (1 - ci_level) / 2
    lo, hi = np.quantile(boots, [tail, 1 - tail])
    return CheckResult(
        name="flagged_split",
        value=float(point),
        passed=bool(lo <= 0 <= hi),
        effect=CheckEffect.REPORT,
        detail=f"flagged minus unflagged gap {point:+.4f}, CI [{lo:+.4f}, {hi:+.4f}]",
    )


def false_alarm_rate(
    texts: Sequence[str], scores: Sequence[float], cfg: ChecksConfig, reps: int, seed: int = 0
) -> CheckResult:
    """Run the drift checks on random halves of one dataset; they should almost never fire."""
    rng = np.random.default_rng(seed)
    texts = np.asarray(texts, dtype=object)
    scores = np.asarray(scores, dtype=float)
    alarms = 0
    for r in range(reps):
        order = rng.permutation(len(texts))
        a, b = order[: len(order) // 2], order[len(order) // 2 :]
        ks = score_drift(scores[a], scores[b], cfg, own_split=True)
        auc = domain_auc(list(texts[a]), list(texts[b]), cfg, own_split=True, seed=seed + r)
        alarms += not (ks.passed and auc.passed)
    rate = alarms / reps
    return CheckResult(
        name="false_alarm_rate",
        value=rate,
        high=cfg.max_false_alarm_rate,
        passed=rate <= cfg.max_false_alarm_rate,
        effect=CheckEffect.INVALID,
        detail=f"{alarms}/{reps} random halves raised drift",
    )
