import hashlib

import pytest

from agent_committee.arms.config import K_SHOTS, MAX_PROMPT_CHARS
from agent_committee.arms.labels import build_label_book
from agent_committee.arms.prompt import PromptArm, build_system_prompt, make_arm
from agent_committee.arms.splits import make_arms_split
from agent_committee.data import load_dev
from agent_committee.llm import LLMClient
from agent_committee.schemas import ArmName

PROMPT_SHA256 = "3d327bfe88ce7d27a6ef3a72eccb8f633b483a4e3915ec5bdd02d6bcffd0ea18"


@pytest.fixture(scope="module")
def split():
    return make_arms_split(load_dev("data"))


@pytest.fixture(scope="module")
def book(split):
    return build_label_book(split.pool)


def fake(text="card_arrival", finish="stop", seen=None):
    def transport(request):
        if seen is not None:
            seen.append(request)
        return {
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }

    return transport


def client(transport):
    return LLMClient(":memory:", phase="build", transport=transport)


def test_real_prompt_fits_the_guard_with_room_for_a_long_query(book):
    # the longest Banking77 query is under 500 characters
    assert len(build_system_prompt(book)) + 500 < MAX_PROMPT_CHARS


def test_prompt_has_k_examples_for_every_label(book):
    lines = build_system_prompt(book).splitlines()
    for label in book.labels:
        assert sum(line.endswith(f" => {label}") for line in lines) == K_SHOTS


def test_prompt_snapshot(book):
    digest = hashlib.sha256(build_system_prompt(book).encode("utf-8")).hexdigest()
    assert digest == PROMPT_SHA256


def test_predict_returns_the_label_with_usage_and_never_sees_gold(split, book):
    seen = []
    arm = make_arm(split.pool, book, client(fake(seen=seen)))
    case = split.tune[0]
    pred = arm.predict(case.id, case.text)
    assert pred.arm == ArmName.PROMPT and pred.pred == "card_arrival"
    assert (pred.prompt_tokens, pred.completion_tokens) == (11, 3)
    assert seen[0]["messages"][1]["content"] == case.text
    assert seen[0]["temperature"] == 0.0


def test_unknown_or_truncated_answers_score_none(split, book):
    arm = make_arm(split.pool, book, client(fake("no idea")))
    assert arm.predict("a", "x").pred is None
    arm = make_arm(split.pool, book, client(fake("card_arrival", finish="length")))
    assert arm.predict("a", "x").pred is None


def test_predict_before_fit_fails(book):
    with pytest.raises(RuntimeError):
        PromptArm(client(fake()), book).predict("a", "x")
