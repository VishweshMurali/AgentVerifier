"""Deterministic pool / tune / eval split of DEV, used by every arm.

pool  (~80% of DEV): what arms may learn from or retrieve.
tune  (~5%): the single place prompts and agent settings may be adjusted.
eval  (~15%): scored once for the DEV report. Never used for tuning.
Any tune/eval case whose normalised text also appears in the pool moves into the pool, so no
text is shared between the pool and the cases that are scored.

Grouped DEV (cases carry `group`, v2 P4): whole groups go to one of pool, tune and eval, and a
duplicate text moves its whole group into the pool, so no group is learned from and scored. Every
label must stay in the pool, and tune and eval must not be empty. Ungrouped DEV takes the original
path unchanged (the v1 layer fingerprint is pinned).
"""

from dataclasses import dataclass

from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from agent_committee.schemas import Case

ARMS_SPLIT_SEED = 20260102
HOLDOUT_FRACTION = 0.2
EVAL_FRACTION_OF_HOLDOUT = 0.75


@dataclass(frozen=True)
class ArmsSplit:
    pool: list[Case]
    tune: list[Case]
    eval: list[Case]

    def counts(self) -> dict[str, int]:
        return {"pool": len(self.pool), "tune": len(self.tune), "eval": len(self.eval)}


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _by_id(cases: list[Case]) -> list[Case]:
    return sorted(cases, key=lambda c: c.id)


def _group_cut(cases: list[Case], fraction: float, seed: int) -> tuple[list[Case], list[Case]]:
    """(rest, cut): whole groups, stratified by label, about `fraction` of the cases in `cut`."""
    folds = StratifiedGroupKFold(
        n_splits=max(2, round(1 / fraction)), shuffle=True, random_state=seed
    )
    rest_idx, cut_idx = next(
        folds.split(range(len(cases)), [c.label for c in cases], [c.group for c in cases])
    )
    return [cases[i] for i in sorted(rest_idx)], [cases[i] for i in sorted(cut_idx)]


def _grouped_split(cases: list[Case], seed: int) -> ArmsSplit:
    pool, holdout = _group_cut(cases, HOLDOUT_FRACTION, seed)
    eval_, tune = _group_cut(holdout, 1 - EVAL_FRACTION_OF_HOLDOUT, seed)
    # a moved group brings its other texts into the pool, which can match texts in further
    # tune/eval groups: repeat until no scored text appears in the pool
    while True:
        pool_texts = {normalize_text(c.text) for c in pool}
        moved = {c.group for c in tune + eval_ if normalize_text(c.text) in pool_texts}
        if not moved:
            break
        pool = pool + [c for c in tune + eval_ if c.group in moved]
        tune = [c for c in tune if c.group not in moved]
        eval_ = [c for c in eval_ if c.group not in moved]
    missing = sorted({c.label for c in cases} - {c.label for c in pool})
    if missing:
        raise ValueError(f"labels missing from the pool after the grouped split: {missing[:5]}")
    if not tune or not eval_:
        raise ValueError("the grouped split left tune or eval empty: too few groups in DEV")
    return ArmsSplit(pool=_by_id(pool), tune=_by_id(tune), eval=_by_id(eval_))


def make_arms_split(dev: list[Case], seed: int = ARMS_SPLIT_SEED) -> ArmsSplit:
    cases = _by_id(dev)
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate case ids in DEV")
    if any(c.group is not None for c in cases):
        if not all(c.group is not None for c in cases):
            raise ValueError("some DEV cases carry a group and some do not")
        return _grouped_split(cases, seed)
    pool, holdout = train_test_split(
        cases, test_size=HOLDOUT_FRACTION, random_state=seed, stratify=[c.label for c in cases]
    )
    tune, eval_ = train_test_split(
        holdout,
        test_size=EVAL_FRACTION_OF_HOLDOUT,
        random_state=seed,
        stratify=[c.label for c in holdout],
    )
    pool_texts = {normalize_text(c.text) for c in pool}
    moved = [c for c in tune + eval_ if normalize_text(c.text) in pool_texts]
    moved_ids = {c.id for c in moved}
    pool = pool + moved
    tune = [c for c in tune if c.id not in moved_ids]
    eval_ = [c for c in eval_ if c.id not in moved_ids]
    return ArmsSplit(pool=_by_id(pool), tune=_by_id(tune), eval=_by_id(eval_))
