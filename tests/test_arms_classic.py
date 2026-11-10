"""v2 P3: the classic rungs, majority floor and Naive Bayes (no LLM, fitted on the pool)."""

import pytest

from agent_committee.arms.classic import MajorityArm, NaiveBayesArm
from agent_committee.schemas import ArmName, Case

POOL = [
    *[
        Case(id=f"c{i}", text=f"where is my card {i} delivery", label="card_arrival")
        for i in range(6)
    ],
    *[Case(id=f"l{i}", text=f"i lost my card {i} stolen", label="lost_card") for i in range(4)],
]


# ---------- majority floor ----------
def test_majority_answers_the_most_common_pool_label_ties_by_sorted_label():
    arm = MajorityArm()
    arm.fit(POOL)
    p = arm.predict("x", "anything")
    assert (p.arm, p.pred, p.prompt_tokens) == (ArmName.MAJORITY, "card_arrival", 0)
    tie = [Case(id="a", text="t", label="zeta"), Case(id="b", text="t", label="alpha")]
    arm.fit(tie)
    assert arm.predict("x", "t").pred == "alpha"


def test_classic_arms_need_fit_first():
    for arm in (MajorityArm(), NaiveBayesArm()):
        with pytest.raises(RuntimeError, match="fit"):
            arm.predict("x", "t")


# ---------- Naive Bayes ----------
def test_naive_bayes_learns_the_pool_and_is_deterministic_and_order_free():
    a, b = NaiveBayesArm(), NaiveBayesArm()
    a.fit(POOL)
    b.fit(list(reversed(POOL)))
    for text in ("my card was stolen", "card delivery status", "zzz"):
        pa, pb = a.predict("x", text), b.predict("x", text)
        assert pa == pb and pa.arm == ArmName.NAIVE_BAYES and pa.prompt_tokens == 0
    assert a.predict("x", "i lost my card stolen").pred == "lost_card"
    assert a.predict("x", "where is my card delivery").pred == "card_arrival"
