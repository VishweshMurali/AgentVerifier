"""The hosted agent (Groq): provider switch, pacing under the limit, rate-limit and refused
tool-call handling. No network: `create` is a fake."""

import httpx
import pytest
from openai import BadRequestError, RateLimitError

from agent_committee import hosted
from agent_committee.arms.agent import MalformedToolCall, _normalize_calls
from agent_committee.llm import LLMClient, LLMUnavailable

REQUEST = {"model": "m", "messages": [{"role": "user", "content": "x" * 350}], "max_tokens": 32}
REPLY = {
    "choices": [{"message": {"role": "assistant", "content": "a"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
}


def _error(cls, status: int, body: dict, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls("refused", response=response, body=body)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_provider_switch(monkeypatch):
    monkeypatch.delenv(hosted.PROVIDER_ENV, raising=False)
    assert hosted.provider() is None
    monkeypatch.setenv(hosted.PROVIDER_ENV, "Ollama")
    assert hosted.provider() is None
    monkeypatch.setenv(hosted.PROVIDER_ENV, "groq")
    assert hosted.provider() == "groq"
    monkeypatch.setenv(hosted.PROVIDER_ENV, "other")
    with pytest.raises(ValueError):
        hosted.provider()


def test_label_and_settings_say_not_pinned(monkeypatch):
    monkeypatch.delenv("GROQ_REASONING", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "some/model")
    assert hosted.label() == "some/model via Groq (hosted, not pinned)"
    assert hosted.settings()["pinned"] is False
    assert hosted.settings()["reasoning_effort"] is None  # only sent to models known to take it
    monkeypatch.delenv("GROQ_MODEL")
    assert hosted.settings()["reasoning_effort"] == "low"
    monkeypatch.setenv("GROQ_REASONING", "none")
    assert hosted.settings()["reasoning_effort"] is None


def test_missing_key_stops(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable):
        hosted.api_key()


def test_pacer_waits_until_the_window_has_room():
    clock = Clock()
    pacer = hosted.Pacer(1000, clock=clock, sleep=clock.sleep)  # 900 usable
    assert pacer.wait(500) == 0.0
    pacer.spend(500)
    clock.now = 10.0
    assert pacer.wait(300) == 0.0
    pacer.spend(300)
    waited = pacer.wait(300)  # 800 + 300 > 900: wait for the first spend to leave the window
    assert waited == pytest.approx(50.0)
    assert clock.now == pytest.approx(60.0)


def test_pacer_lets_an_oversized_request_through_an_empty_window():
    clock = Clock()
    pacer = hosted.Pacer(100, clock=clock, sleep=clock.sleep)
    assert pacer.wait(5000) == 0.0


def test_transport_sends_low_reasoning_and_records_usage(monkeypatch):
    monkeypatch.delenv("GROQ_REASONING", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return REPLY

    clock = Clock()
    pacer = hosted.Pacer(8000, clock=clock, sleep=clock.sleep)
    out = hosted.groq_transport(pacer=pacer, create=create, sleep=clock.sleep)(REQUEST)
    assert out == REPLY
    assert seen["reasoning_effort"] == "low"
    assert pacer._used(clock.now) == 105


def test_rate_limit_waits_for_the_named_time_then_answers():
    calls = []

    def create(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _error(RateLimitError, 429, {}, {"retry-after": "7"})
        return REPLY

    clock = Clock()
    pacer = hosted.Pacer(8000, clock=clock, sleep=clock.sleep)
    out = hosted.groq_transport(pacer=pacer, create=create, sleep=clock.sleep)(REQUEST)
    assert out == REPLY and len(calls) == 2
    assert 7.0 in clock.slept


def test_daily_limit_stops_the_run():
    def create(**kwargs):
        raise _error(RateLimitError, 429, {}, {"retry-after": "3600"})

    clock = Clock()
    pacer = hosted.Pacer(8000, clock=clock, sleep=clock.sleep)
    with pytest.raises(LLMUnavailable, match="used up"):
        hosted.groq_transport(pacer=pacer, create=create, sleep=clock.sleep)(REQUEST)
    assert clock.slept == []


def test_refused_tool_call_becomes_a_malformed_call_with_tokens_charged():
    body = {"error": {"code": "tool_use_failed", "failed_generation": '{"query": "unfinished'}}

    def create(**kwargs):
        raise _error(BadRequestError, 400, body)

    clock = Clock()
    pacer = hosted.Pacer(8000, clock=clock, sleep=clock.sleep)
    transport = hosted.groq_transport(pacer=pacer, create=create, sleep=clock.sleep)
    with LLMClient(":memory:", phase="build", model="m", transport=transport) as client:
        result = client.chat(REQUEST["messages"], arm="agent", max_tokens=32)
    assert result.completion_tokens == 32 and result.prompt_tokens > 0
    with pytest.raises(MalformedToolCall):  # the agent scores it wrong, like any malformed call
        _normalize_calls(result.message["tool_calls"], 0)


def test_other_bad_requests_are_not_swallowed():
    def create(**kwargs):
        raise _error(BadRequestError, 400, {"error": {"code": "context_length_exceeded"}})

    clock = Clock()
    pacer = hosted.Pacer(8000, clock=clock, sleep=clock.sleep)
    with pytest.raises(BadRequestError):
        hosted.groq_transport(pacer=pacer, create=create, sleep=clock.sleep)(REQUEST)


def test_the_commit_comes_from_the_host_when_there_is_no_git_folder(tmp_path, monkeypatch):
    """A deployed image has no .git: the commit the host names is recorded instead."""
    import subprocess

    from agent_committee import quick

    def no_git(repo):
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr("agent_committee.runner.git_commit", no_git)
    monkeypatch.delenv("SOURCE_COMMIT", raising=False)
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc1234\n")
    assert quick.source_commit(tmp_path) == "abc1234"
    monkeypatch.delenv("RENDER_GIT_COMMIT")
    with pytest.raises(subprocess.CalledProcessError):  # never an invented commit
        quick.source_commit(tmp_path)
