"""v2 P3: the cascade, derived from linear and agent predictions (no model calls)."""

import pytest

from agent_committee.arms.cascade import cascade_predictions
from agent_committee.schemas import ArmName, ArmPrediction


def pred(case_id, label, arm=ArmName.LINEAR, pt=0, ct=0, lat=0.0, **kw):
    return ArmPrediction(
        case_id=case_id,
        arm=arm,
        pred=label,
        prompt_tokens=pt,
        completion_tokens=ct,
        latency_s=lat,
        **kw,
    )


# ---------- the cascade ----------
def test_cascade_takes_linear_when_confident_and_the_agent_when_flagged():
    linear = [pred("1", "a"), pred("2", "b"), pred("3", "c")]
    agent = [
        pred("1", "x", ArmName.AGENT, 100, 5, 1.0),
        pred("2", "y", ArmName.AGENT, 200, 6, 2.0, own_tool_calls=1),
        pred("3", None, ArmName.AGENT, 300, 7, 3.0, malformed=True),
    ]
    out = cascade_predictions(linear, agent, [False, True, True])
    assert [p.pred for p in out] == ["a", "y", None]
    assert all(p.arm == ArmName.CASCADE for p in out)
    assert [(p.prompt_tokens, p.completion_tokens, p.latency_s) for p in out] == [
        (0, 0, 0.0),
        (200, 6, 2.0),
        (300, 7, 3.0),
    ]
    assert (out[1].own_tool_calls, out[2].malformed) == (1, True)


def test_cascade_refuses_misaligned_inputs():
    with pytest.raises(ValueError, match="aligned"):
        cascade_predictions([pred("1", "a")], [], [False])
    with pytest.raises(ValueError, match="case order"):
        cascade_predictions([pred("1", "a")], [pred("2", "a", ArmName.AGENT)], [True])
