import pytest

from agent_committee.arms.tfidf import TfidfArm
from agent_committee.schemas import ArmName, Case


def _pool():
    return [Case(id=f"a{i}", text=f"lost card number {i}", label="lost") for i in range(10)] + [
        Case(id=f"b{i}", text=f"exchange rate currency {i}", label="fx") for i in range(10)
    ]


def test_fit_predict_and_no_label_leak():
    arm = TfidfArm()
    arm.fit(_pool())
    p = arm.predict("q1", "my card is lost")
    assert p.pred == "lost" and p.arm == ArmName.TFIDF and p.case_id == "q1"
    assert arm.predict("q2", "what currency rate").pred == "fx"


def test_deterministic_and_requires_fit():
    a, b = TfidfArm(), TfidfArm()
    a.fit(_pool())
    b.fit(list(reversed(_pool())))
    assert a.predict("q", "card rate").pred == b.predict("q", "card rate").pred
    with pytest.raises(RuntimeError):
        TfidfArm().predict("q", "x")
