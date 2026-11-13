from verifier_server import guide

FACTS = (
    "Your agent got 3 of 95 wrong. Monthly cost: agent $4,747, Linear $1,579. Right: 97% vs 95%."
)


def test_numbers_are_normalised():
    assert guide.numbers("costs $4,747 a month, 97% right, gap −2.10") == {"4747", "97", "-2.1"}


def test_a_reply_with_only_known_numbers_passes():
    ok, bad = guide.grounded(
        "Your agent missed 3 of 95; Linear costs $1,579 a month.", FACTS, "why?"
    )
    assert ok and bad == []


def test_an_invented_number_is_caught():
    ok, bad = guide.grounded("Linear would save you about $3,200 a month.", FACTS, "how much?")
    assert not ok and bad == ["3200"]


def test_numbers_from_the_readers_question_are_allowed():
    ok, _ = guide.grounded("At $8 per mistake the gap grows.", FACTS, "what if a mistake costs $8?")
    assert ok


def test_the_model_reply_is_used_when_grounded():
    out = guide.guide_reply(
        {
            "question": "why did it lose?",
            "description": "refund bot",
            "facts": FACTS,
            "fallback": "plain",
        },
        call=lambda body: "Your refund bot slipped on 3 of 95 messages, so Linear wins.",
    )
    assert out["worded_by"] == "model" and out["reply"].startswith("Your refund bot")


def test_an_ungrounded_reply_falls_back_to_the_plain_answer():
    out = guide.guide_reply(
        {"question": "why?", "description": "", "facts": FACTS, "fallback": "plain"},
        call=lambda body: "It was wrong 12 times.",
    )
    assert out == {"reply": "plain", "worded_by": "code", "checked": True, "rejected": ["12"]}


def test_model_errors_fall_back_too():
    def boom(body):
        raise TimeoutError("slow")

    out = guide.guide_reply({"question": "q", "facts": FACTS, "fallback": "plain"}, call=boom)
    assert out["reply"] == "plain" and out["worded_by"] == "code"


def test_the_call_is_deterministic_and_bounded():
    seen = {}
    guide.guide_reply(
        {"question": "q", "facts": FACTS, "fallback": "p"}, call=lambda b: seen.update(b) or "ok"
    )
    assert seen["temperature"] == 0 and seen["seed"] == 0 and seen["max_tokens"] <= 300


def test_a_count_must_match_its_total():
    facts = (
        "refund not showing up: 20 messages, agent wrong on 2 of 20. "
        "Overall 3 of 95 wrong, Linear 1 of 95."
    )
    ok, bad = guide.grounded("It got 1 out of 20 wrong on refund not showing up.", facts, "why?")
    assert not ok and bad == ["1 of 20"]
    ok, _ = guide.grounded("It got 2 of 20 wrong there, and 3 of 95 overall.", facts, "why?")
    assert ok
