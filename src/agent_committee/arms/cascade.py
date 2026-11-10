"""The cascade (v2 P3): the linear layer answers confident cases, the agent only Null-flagged ones.

It is derived from predictions the linear and agent arms already made, case by case, and never
calls a model itself: the LLM cache is keyed on the arm name, so a separate cascade arm would
re-ask the model and could get a different answer than the agent arm did. Its tokens and latency
are the agent's on flagged cases only (the linear layer costs nothing).

The cascade is reported next to the other arms; it is not the agent the verdict is about.
"""

from collections.abc import Sequence

from agent_committee.schemas import ArmName, ArmPrediction


def cascade_predictions(
    linear: Sequence[ArmPrediction],
    agent: Sequence[ArmPrediction],
    flagged: Sequence[bool],
) -> list[ArmPrediction]:
    """Per case: the agent's prediction (and cost) when flagged, else the linear one at no cost."""
    if not len(linear) == len(agent) == len(flagged):
        raise ValueError("linear, agent and flagged must be aligned case by case")
    out = []
    for lin, agt, flag in zip(linear, agent, flagged, strict=True):
        if lin.case_id != agt.case_id:
            raise ValueError(f"case order differs: {lin.case_id} vs {agt.case_id}")
        src = agt if flag else lin
        out.append(
            ArmPrediction(
                case_id=lin.case_id,
                arm=ArmName.CASCADE,
                pred=src.pred,
                prompt_tokens=agt.prompt_tokens if flag else 0,
                completion_tokens=agt.completion_tokens if flag else 0,
                latency_s=agt.latency_s if flag else 0.0,
                own_tool_calls=agt.own_tool_calls if flag else 0,
                sent_back=agt.sent_back if flag else False,
                malformed=agt.malformed if flag else False,
                too_long=agt.too_long if flag else False,
            )
        )
    return out
