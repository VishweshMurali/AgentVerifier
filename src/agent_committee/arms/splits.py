"""Deterministic pool / tune / eval split of DEV, used by every arm.

pool  (~80% of DEV): what arms may learn from or retrieve.
tune  (~5%): the single place prompts and agent settings may be adjusted.
eval  (~15%): scored once for the DEV report. Never used for tuning.
Any tune/eval case whose normalised text also appears in the pool moves into the pool, so no
text is shared between the pool and the cases that are scored.
"""

from dataclasses import dataclass

from sklearn.model_selection import train_test_split

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


def make_arms_split(dev: list[Case], seed: int = ARMS_SPLIT_SEED) -> ArmsSplit:
    cases = _by_id(dev)
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate case ids in DEV")
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
