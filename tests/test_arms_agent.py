import json
from types import SimpleNamespace

import pytest

from agent_committee.arms import parsing
from agent_committee.arms.agent import (
    SEND_BACK,
    AgentArm,
    build_system_prompt,
    make_arm,
)
from agent_committee.arms.agent_tools import READ_LABEL, SEARCH, TOOLS, ToolBox
from agent_committee.arms.config import AGENT_MAX_TOOL_TURNS, MAX_PROMPT_CHARS
from agent_committee.arms.labels import build_label_book
from agent_committee.arms.parsing import PromptTooLong, guarded_chat, prompt_chars
from agent_committee.arms.retrieval import Retriever
from agent_committee.llm import LLMClient, LLMUnavailable
from agent_committee.schemas import ArmName, Case

LABELS = ("card_arrival", "lost_or_stolen_card", "card_delivery_estimate")
POOL = [Case(id=f"{x}{i}", text=f"{x} sample {i}", label=x) for x in LABELS for i in range(3)]
BOOK = build_label_book(POOL)
SHORTLIST = LABELS[:2]
LAYER = SimpleNamespace(decide=lambda text: SimpleNamespace(shortlist=SHORTLIST))


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
    return make_arm(POOL, BOOK, LAYER, LLMClient(":memory:", phase="build", transport=transport))


SEARCHED = reply(calls=[call(SEARCH, {"query": "card"})])


def test_prompt_gives_the_shortlist_not_the_full_label_list():
    prompt = build_system_prompt(SHORTLIST)
    assert all(label in prompt for label in SHORTLIST)
    assert LABELS[2] not in prompt


def test_tool_then_answer_sums_usage_and_counts_own_tool_use():
    script = Script(SEARCHED, reply("card_arrival"))
    arm = arm_with(script)
    pred = arm.predict("a", "x")
    assert pred.arm == ArmName.AGENT and pred.pred == "card_arrival"
    assert (pred.prompt_tokens, pred.completion_tokens) == (20, 4)
    assert arm.stats["own_tool_cases"] == 1 and arm.stats["sent_back_cases"] == 0
    assert (pred.own_tool_calls, pred.sent_back, pred.malformed) == (1, False, False)
    assert script.requests[1]["messages"][-1]["role"] == "tool"


def test_no_search_runs_unless_the_model_asks():
    script = Script(reply("card_arrival"), SEARCHED, reply("card_arrival"))
    arm_with(script).predict("a", "x")
    assert not any(m["role"] == "tool" for m in script.requests[0]["messages"])


def test_answer_before_any_tool_is_sent_back_once():
    script = Script(reply("card_arrival"), SEARCHED, reply("card_arrival"))
    arm = arm_with(script)
    assert arm.predict("a", "x").pred == "card_arrival"
    assert script.requests[1]["messages"][-1]["content"] == SEND_BACK
    assert arm.stats["sent_back_cases"] == 1 and arm.stats["own_tool_cases"] == 0


def test_second_no_tool_answer_is_taken_as_given():
    script = Script(reply("card_arrival"), reply("lost_or_stolen_card"))
    assert arm_with(script).predict("a", "x").pred == "lost_or_stolen_card"
    assert len(script.requests) == 2


def test_label_outside_the_shortlist_is_allowed():
    script = Script(SEARCHED, reply("card_delivery_estimate"))
    assert arm_with(script).predict("a", "x").pred == "card_delivery_estimate"


def test_invalid_answer_after_tools_goes_to_decide_without_tools():
    script = Script(SEARCHED, reply("not sure"), reply("card_arrival"))
    assert arm_with(script).predict("a", "x").pred == "card_arrival"
    assert "tools" not in script.requests[-1]


def test_model_calls_are_capped_and_the_last_must_answer():
    looping = [reply(calls=[call(SEARCH, {"query": f"q{i}"})]) for i in range(AGENT_MAX_TOOL_TURNS)]
    script = Script(*looping, reply("card_arrival"))
    arm = arm_with(script)
    assert arm.predict("a", "x").pred == "card_arrival"
    assert len(script.requests) == AGENT_MAX_TOOL_TURNS + 1
    assert "tools" not in script.requests[-1]
    assert arm.stats["model_calls"] == AGENT_MAX_TOOL_TURNS + 1


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
    assert sum(m["role"] == "tool" for m in script.requests[1]["messages"]) == 3


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "x"},
        {"function": "nope"},
        {"function": {"name": SEARCH}},
        {"function": {"arguments": "{}"}},
        {"function": {"name": SEARCH, "arguments": {"query": "not a string"}}},
    ],
)
def test_malformed_tool_call_scores_wrong_and_does_not_stop_the_run(bad):
    script = Script(reply(calls=[bad]))
    arm = arm_with(script)
    pred = arm.predict("a", "x")
    assert pred.pred is None and pred.malformed
    assert arm.stats["malformed_cases"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"function": {"name": SEARCH, "arguments": "{bad"}},
        {"function": {"name": SEARCH, "arguments": "[1]"}},
        {"function": {"name": SEARCH, "arguments": json.dumps({"wrong": "x"})}},
        {"function": {"name": READ_LABEL, "arguments": json.dumps({"label": 3})}},
        {"function": {"name": "delete_all", "arguments": "{}"}},
    ],
)
def test_bad_arguments_or_unknown_tool_are_malformed_and_no_tool_runs(bad):
    script = Script(reply(calls=[call(SEARCH, {"query": "ok"}), bad]))
    arm = arm_with(script)
    pred = arm.predict("a", "x")
    assert pred.pred is None and pred.malformed and pred.own_tool_calls == 0
    assert len(script.requests) == 1 and arm.stats["own_tool_cases"] == 0


def test_malformed_call_past_the_per_turn_cap_is_still_malformed():
    calls = [call(SEARCH, {"query": f"q{i}"}, f"id{i}") for i in range(3)]
    script = Script(reply(calls=[*calls, {"function": "nope"}]))
    pred = arm_with(script).predict("a", "x")
    assert pred.pred is None and pred.malformed


def test_tool_call_in_a_truncated_reply_is_malformed():
    script = Script(reply(calls=[call(SEARCH, {"query": "card"})], finish="length"))
    pred = arm_with(script).predict("a", "x")
    assert pred.pred is None and pred.malformed


def test_tool_call_in_the_no_tools_decide_turn_is_malformed():
    looping = [reply(calls=[call(SEARCH, {"query": f"q{i}"})]) for i in range(AGENT_MAX_TOOL_TURNS)]
    last = reply("card_arrival", calls=[call(SEARCH, {"query": "more"})])
    arm = arm_with(Script(*looping, last))
    pred = arm.predict("a", "x")
    assert pred.pred is None and pred.malformed
    assert pred.own_tool_calls == AGENT_MAX_TOOL_TURNS
    assert pred.prompt_tokens == 10 * (AGENT_MAX_TOOL_TURNS + 1)


def test_reply_the_llm_layer_rejects_is_malformed_and_keeps_usage():
    broken = reply()
    broken["choices"][0]["message"]["tool_calls"] = "not a list"
    arm = arm_with(Script(SEARCHED, broken))
    pred = arm.predict("a", "x")
    assert pred.pred is None and pred.malformed and pred.own_tool_calls == 1
    assert pred.prompt_tokens >= 10 and arm.stats["malformed_cases"] == 1


def test_truncated_final_answer_is_none():
    script = Script(SEARCHED, reply("card_arrival", finish="length"), reply("x", finish="length"))
    assert arm_with(script).predict("a", "x").pred is None


def test_llm_outage_stops_the_run():
    def down(request):
        raise LLMUnavailable("down")

    with pytest.raises(LLMUnavailable):
        arm_with(down).predict("a", "x")


def test_oversized_prompt_is_never_sent_and_scores_wrong():
    script = Script()
    arm = arm_with(script)
    pred = arm.predict("a", "x" * (MAX_PROMPT_CHARS + 1))
    assert pred.pred is None and pred.too_long
    assert script.requests == []
    assert arm.stats["too_long_cases"] == 1


def test_guard_still_refuses_the_call_itself():
    with pytest.raises(PromptTooLong):
        guarded_chat(
            None,
            [{"role": "user", "content": "x" * (MAX_PROMPT_CHARS + 1)}],
            arm="agent",
            max_tokens=1,
        )


def test_too_long_after_a_tool_turn_keeps_the_usage_spent(monkeypatch):
    first = [
        {"role": "system", "content": build_system_prompt(SHORTLIST)},
        {"role": "user", "content": "x"},
    ]
    monkeypatch.setattr(parsing, "MAX_PROMPT_CHARS", prompt_chars(first, TOOLS))
    script = Script(SEARCHED)
    arm = arm_with(script)
    pred = arm.predict("a", "x")
    assert pred.pred is None and pred.too_long and not pred.malformed
    assert (pred.prompt_tokens, pred.completion_tokens, pred.own_tool_calls) == (10, 2, 1)
    assert len(script.requests) == 1 and arm.stats["too_long_cases"] == 1


def test_predict_before_fit_fails():
    arm = AgentArm(LLMClient(":memory:", phase="build", transport=Script()), BOOK, LAYER)
    with pytest.raises(RuntimeError):
        arm.predict("a", "x")


def test_search_returns_examples_for_shortlisted_labels_best_first():
    box = ToolBox(Retriever(POOL), BOOK, LAYER)
    lines = box.run(SEARCH, json.dumps({"query": "lost_or_stolen_card sample"})).splitlines()
    assert lines[0].endswith("=> card_arrival")
    assert {line.split(" => ")[1] for line in lines} == set(SHORTLIST)


def test_read_label_resolves_and_suggests_closest_real_labels():
    box = ToolBox(Retriever(POOL), BOOK, LAYER)
    assert "card_arrival" in box.run(READ_LABEL, json.dumps({"label": "CARD ARRIVAL"}))
    unknown = box.run(READ_LABEL, json.dumps({"label": "card_delivery_time"}))
    assert unknown.startswith("Unknown label")
    assert "card_delivery_estimate" in unknown.split("Closest labels: ")[1].split(", ")[0]


def test_bad_tool_calls_return_readable_errors():
    box = ToolBox(Retriever(POOL), BOOK, LAYER)
    assert box.run(SEARCH, "{bad").startswith("Invalid JSON")
    assert box.run(SEARCH, "[1]").startswith("Invalid arguments")
    assert box.run("other", "{}").startswith("Invalid call")
    assert box.run(SEARCH, json.dumps({"wrong": 1})).startswith("Invalid call")
    assert {t["function"]["name"] for t in TOOLS} == {SEARCH, READ_LABEL}
