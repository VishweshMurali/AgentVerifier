import hashlib
from types import SimpleNamespace

import pytest

from agent_committee.arms.config import K_SHOTS, MAX_PROMPT_CHARS, SHORTLIST_K
from agent_committee.arms.labels import build_label_book
from agent_committee.arms.linear import make_layer
from agent_committee.arms.prompt import build_system_prompt, make_arm
from agent_committee.arms.splits import make_arms_split
from agent_committee.data import load_dev
from agent_committee.llm import LLMClient
from agent_committee.schemas import ArmName

PROMPT_SHA256 = "739ada12ecfd47d95555bde6f2e72b564f28e22a4e2ac47c92300d72636b2728"


@pytest.fixture(scope="module")
def split():
    return make_arms_split(load_dev("data"))


@pytest.fixture(scope="module")
def book(split):
    return build_label_book(split.pool)


@pytest.fixture(scope="module")
def layer(book):
    shortlist = book.labels[:SHORTLIST_K]
    return SimpleNamespace(decide=lambda text: SimpleNamespace(shortlist=shortlist))


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


def test_real_layer_prompt_fits_the_guard(split, book):
    real = make_layer(split.pool)
    case = max(split.tune, key=lambda c: len(c.text))
    prompt = build_system_prompt(book, real.decide(case.text).shortlist)
    assert len(prompt) + len(case.text) < MAX_PROMPT_CHARS


def test_prompt_has_k_examples_for_each_candidate_and_no_other_label(book):
    shortlist = book.labels[:SHORTLIST_K]
    lines = build_system_prompt(book, shortlist).splitlines()
    for label in shortlist:
        assert sum(line.endswith(f" => {label}") for line in lines) == K_SHOTS
    assert not any(line.endswith(f" => {book.labels[-1]}") for line in lines)


def test_prompt_snapshot(book):
    prompt = build_system_prompt(book, book.labels[:SHORTLIST_K])
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == PROMPT_SHA256


def test_predict_returns_the_label_with_usage_and_never_sees_gold(split, book, layer):
    seen = []
    arm = make_arm(split.pool, book, layer, client(fake(seen=seen)))
    case = split.tune[0]
    pred = arm.predict(case.id, case.text)
    assert pred.arm == ArmName.PROMPT and pred.pred == "card_arrival"
    assert (pred.prompt_tokens, pred.completion_tokens) == (11, 3)
    assert seen[0]["messages"][1]["content"] == case.text
    assert seen[0]["temperature"] == 0.0


def test_oversized_prompt_is_never_sent_and_scores_wrong(split, book, layer):
    seen = []
    arm = make_arm(split.pool, book, layer, client(fake(seen=seen)))
    pred = arm.predict("a", "x" * (MAX_PROMPT_CHARS + 1))
    assert pred.pred is None and pred.too_long and seen == []


def test_unknown_or_truncated_answers_score_none(split, book, layer):
    arm = make_arm(split.pool, book, layer, client(fake("no idea")))
    assert arm.predict("a", "x").pred is None
    arm = make_arm(split.pool, book, layer, client(fake("card_arrival", finish="length")))
    assert arm.predict("a", "x").pred is None


def test_malformed_reply_scores_none_and_keeps_usage(split, book, layer):
    arm = make_arm(split.pool, book, layer, client(fake(text=["not", "a string"])))
    pred = arm.predict("a", "x")
    assert pred.pred is None and (pred.prompt_tokens, pred.completion_tokens) == (11, 3)
