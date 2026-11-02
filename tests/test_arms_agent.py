import json

import pytest

from agent_committee.arms.agent import AgentArm, build_system_prompt, make_arm
from agent_committee.arms.agent_tools import READ_LABEL, SEARCH, TOOLS, ToolBox
from agent_committee.arms.config import AGENT_MAX_TOOL_TURNS, MAX_PROMPT_CHARS
from agent_committee.arms.labels import build_label_book
from agent_committee.arms.parsing import PromptTooLong
from agent_committee.arms.retrieval import Retriever
from agent_committee.llm import LLMClient, LLMUnavailable
from agent_committee.schemas import ArmName, Case

POOL = [
    Case(id=f"{label}{i}", text=f"{label} sample {i}", label=label)
    for label in ("card_arrival", "lost_or_stolen_card")
    for i in range(3)
]
BOOK = build_label_book(POOL)


def reply(content="", calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


def call(name, args, id_="call_random"):
    return {
        "id": id_,
        "type": "function",
        "index": 0,
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class Script:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(json.dumps(request)))
        return self.responses.pop(0)


def arm_with(transport):
    return make_arm(POOL, BOOK, LLMClient(":memory:", phase="build", transport=transport))


def test_direct_valid_answer_costs_one_call():
    script = Script(reply("card_arrival"))
    pred = arm_with(script).predict("a", "where is my card")
    assert pred.arm == ArmName.AGENT and pred.pred == "card_arrival"
    assert len(script.requests) == 1


def test_first_search_always_runs_and_costs_no_model_call():
    script = Script(reply("card_arrival"))
    arm_with(script).predict("a", "card_arrival sample")
    messages = script.requests[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["id"] == "call_0"
    assert messages[3]["tool_call_id"] == "call_0" and "=> card_arrival" in messages[3]["content"]


def test_model_tool_ids_continue_after_the_first_search():
    script = Script(reply(calls=[call(SEARCH, {"query": "q"}, "x")]), reply("card_arrival"))
    arm_with(script).predict("a", "x")
    assert script.requests[1]["messages"][-1]["tool_call_id"] == "call_1"


def test_invalid_direct_answer_goes_to_decide_without_tools():
    script = Script(reply("hmm, not sure"), reply("lost_or_stolen_card"))
    assert arm_with(script).predict("a", "x").pred == "lost_or_stolen_card"
    assert "tools" not in script.requests[1]


def test_tool_loop_then_answer_sums_usage():
    script = Script(reply(calls=[call(SEARCH, {"query": "card"})]), reply("card_arrival"))
    pred = arm_with(script).predict("a", "x")
    assert pred.pred == "card_arrival"
    assert (pred.prompt_tokens, pred.completion_tokens) == (20, 4)
    tool_message = script.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and "=> " in tool_message["content"]


def test_tool_turns_are_capped_and_decide_has_no_tools():
    looping = [reply(calls=[call(SEARCH, {"query": f"q{i}"})]) for i in range(AGENT_MAX_TOOL_TURNS)]
    script = Script(*looping, reply("card_arrival"))
    assert arm_with(script).predict("a", "x").pred == "card_arrival"
    assert len(script.requests) == AGENT_MAX_TOOL_TURNS + 1
    assert "tools" not in script.requests[-1]
    assert all("tools" in r for r in script.requests[:-1])


def test_random_model_tool_ids_are_normalised_for_stable_cache_keys():
    a = Script(reply(calls=[call(SEARCH, {"query": "q"}, "call_aaa")]), reply("card_arrival"))
    b = Script(reply(calls=[call(SEARCH, {"query": "q"}, "call_zzz")]), reply("card_arrival"))
    arm_with(a).predict("a", "x")
    arm_with(b).predict("a", "x")
    assert a.requests[1]["messages"] == b.requests[1]["messages"]


def test_calls_per_turn_are_capped():
    many = [call(SEARCH, {"query": f"q{i}"}, f"id{i}") for i in range(9)]
    script = Script(reply(calls=many), reply("card_arrival"))
    arm_with(script).predict("a", "x")
    # the deterministic first search, plus at most 3 from the model
    assert sum(m["role"] == "tool" for m in script.requests[1]["messages"]) == 4


def test_truncated_final_answer_is_none():
    script = Script(reply("nope"), reply("card_arrival", finish="length"))
    assert arm_with(script).predict("a", "x").pred is None


def test_llm_outage_stops_the_run():
    def down(request):
        raise LLMUnavailable("down")

    with pytest.raises(LLMUnavailable):
        arm_with(down).predict("a", "x")


def test_oversized_prompt_is_refused():
    script = Script()
    with pytest.raises(PromptTooLong):
        arm_with(script).predict("a", "x" * (MAX_PROMPT_CHARS + 1))
    assert script.requests == []


def test_predict_before_fit_fails():
    arm = AgentArm(LLMClient(":memory:", phase="build", transport=Script()), BOOK)
    with pytest.raises(RuntimeError):
        arm.predict("a", "x")


def test_system_prompt_lists_every_label():
    prompt = build_system_prompt(BOOK)
    assert all(label in prompt for label in BOOK.labels)


def test_tools_box_outputs_and_bad_calls():
    box = ToolBox(Retriever(POOL), BOOK)
    assert "=> card_arrival" in box.run(SEARCH, json.dumps({"query": "card arrival"}))
    assert "card_arrival" in box.run(READ_LABEL, json.dumps({"label": "CARD ARRIVAL"}))
    assert box.run(READ_LABEL, json.dumps({"label": "nope"})).startswith("Unknown label")
    assert box.run(SEARCH, "{bad").startswith("Invalid JSON")
    assert box.run(SEARCH, "[1]").startswith("Invalid arguments")
    assert box.run("other", "{}").startswith("Invalid call")
    assert box.run(SEARCH, json.dumps({"wrong": 1})).startswith("Invalid call")
    assert {t["function"]["name"] for t in TOOLS} == {SEARCH, READ_LABEL}
