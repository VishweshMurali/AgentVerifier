import dataclasses

import pytest

from agent_committee.arms import config
from agent_committee.arms.linear import Decision, LinearArm, LinearLayer, make_layer
from agent_committee.schemas import ArmName, Case

TOPICS = {
    "lost": "lost card number",
    "fx": "exchange rate currency",
    "pin": "forgot my pin code",
    "fee": "transfer fee charged",
    "atm": "cash machine withdrawal",
    "top": "top up account balance",
}


def _pool():
    return [
        Case(id=f"{label}{i}", text=f"{text} {i}", label=label)
        for label, text in TOPICS.items()
        for i in range(8)
    ]


def test_decide_returns_choice_score_and_top_k_shortlist():
    d = make_layer(_pool()).decide("my card is lost")
    assert isinstance(d, Decision)
    assert d.choice == "lost" == d.shortlist[0]
    assert len(d.shortlist) == config.SHORTLIST_K == len(set(d.shortlist))
    assert set(d.shortlist) <= set(TOPICS)
    assert 0.0 < d.score <= 1.0
    assert d.null is False


def test_decision_is_frozen():
    d = make_layer(_pool()).decide("my card is lost")
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.choice = "fx"


def test_tied_scores_break_by_label_name():
    # two identical classes in every feature, so their probabilities are exactly tied
    pool = [Case(id=f"{x}{i}", text=f"same words {i}", label=x) for x in "ba" for i in range(4)]
    layer = LinearLayer(shortlist_k=2)
    layer.fit(pool)
    assert layer.decide("same words").shortlist == ("a", "b")


def test_null_is_score_strictly_below_threshold():
    pool = _pool()
    score = make_layer(pool).decide("my card is lost").score

    at = LinearLayer(threshold=score)
    at.fit(pool)
    assert at.decide("my card is lost").null is False

    above = LinearLayer(threshold=score + 1e-6)
    above.fit(pool)
    d = above.decide("my card is lost")
    assert d.null is True and d.choice == "lost"  # a flag changes neither choice nor shortlist


def test_fit_is_once_only_and_required():
    layer = make_layer(_pool())
    with pytest.raises(RuntimeError, match="frozen"):
        layer.fit(_pool())
    with pytest.raises(RuntimeError, match="fit"):
        LinearLayer().decide("x")
    with pytest.raises(RuntimeError, match="fit"):
        LinearLayer().fingerprint()


def test_fingerprint_is_stable_across_rebuild_and_input_order():
    a = make_layer(_pool())
    b = make_layer(list(reversed(_pool())))
    assert a.fingerprint() == b.fingerprint()
    assert a.decide("card rate") == b.decide("card rate")


def test_fingerprint_changes_with_data_and_with_settings():
    base = make_layer(_pool()).fingerprint()
    assert make_layer(_pool()[:-1]).fingerprint() != base
    other = LinearLayer(threshold=0.5)
    other.fit(_pool())
    assert other.fingerprint() != base


def test_arm_answers_from_the_layer_with_zero_tokens():
    arm = LinearArm(make_layer(_pool()))
    p = arm.predict("q1", "my card is lost")
    assert (p.case_id, p.arm, p.pred) == ("q1", ArmName.LINEAR, "lost")
    assert p.prompt_tokens == 0 and p.completion_tokens == 0


def test_arm_answers_with_its_choice_even_when_flagged():
    layer = LinearLayer(threshold=1.1)
    layer.fit(_pool())
    assert layer.decide("my card is lost").null is True
    assert LinearArm(layer).predict("q", "my card is lost").pred == "lost"


def test_arm_fit_fills_an_empty_layer_and_leaves_a_frozen_one():
    empty = LinearLayer()
    LinearArm(empty).fit(_pool())
    assert empty.fitted

    frozen = make_layer(_pool())
    before = frozen.fingerprint()
    LinearArm(frozen).fit(_pool()[:6])
    assert frozen.fingerprint() == before
