import pytest

from agent_committee.arms.config import MAX_PROMPT_CHARS
from agent_committee.arms.labels import build_label_book
from agent_committee.arms.parsing import (
    PromptTooLong,
    guarded_chat,
    parse_answer,
    prompt_chars,
)
from agent_committee.llm import LLMResult
from agent_committee.schemas import Case

BOOK = build_label_book(
    [
        Case(id=f"{label}{i}", text=f"{label} word{i}", label=label)
        for label in ("card_arrival", "Refund_not_showing_up", "reverted_card_payment?")
        for i in range(2)
    ]
)


def result(text, finish="stop"):
    return LLMResult({"content": text}, 1, 1, 0.0, False, 1, finish)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("card_arrival", "card_arrival"),
        ("  `card_arrival`\n", "card_arrival"),
        ("CARD ARRIVAL", "card_arrival"),
        ("refund_not_showing_up", "Refund_not_showing_up"),
        ("reverted_card_payment", "reverted_card_payment?"),
        ("", None),
        ("   ", None),
        ("not_a_label", None),
        ("I think card_arrival", None),
        ("card_arrival\nbecause it arrived", None),
    ],
)
def test_parse_answer(text, expected):
    assert parse_answer(result(text), BOOK) == expected


def test_truncated_output_is_unparseable_even_if_it_looks_valid():
    assert parse_answer(result("card_arrival", finish="length"), BOOK) is None


def test_prompt_chars_counts_tools_and_replayed_calls():
    base = [{"role": "user", "content": "abc"}]
    assert prompt_chars(base) == 3
    assert prompt_chars(base, [{"a": 1}]) > 3
    replay = {"role": "assistant", "content": None, "tool_calls": [{"a": 1}]}
    assert prompt_chars([*base, replay]) > 3


class Client:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return result("card_arrival")


def test_guard_refuses_before_any_call():
    client = Client()
    too_long = [{"role": "user", "content": "x" * (MAX_PROMPT_CHARS + 1)}]
    with pytest.raises(PromptTooLong):
        guarded_chat(client, too_long, arm="prompt", max_tokens=8)
    assert client.calls == 0


def test_guard_allows_a_prompt_at_the_limit():
    client = Client()
    ok = [{"role": "user", "content": "x" * MAX_PROMPT_CHARS}]
    guarded_chat(client, ok, arm="prompt", max_tokens=8)
    assert client.calls == 1
