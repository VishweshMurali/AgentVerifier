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


TOOLS = [{"type": "function", "function": {"name": "search_examples"}}]
FIRST = {**REQUEST, "tools": TOOLS}
AFTER_TOOL = {
    **FIRST,
    "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "..."},
    ],
}
SENT_BACK = {
    **FIRST,
    "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "my answer"},
        {"role": "user", "content": "Do not answer yet."},
    ],
}


def _sent(request, monkeypatch, first_tool=None):
    monkeypatch.delenv("GROQ_REASONING", raising=False)
    if first_tool is None:
        monkeypatch.delenv(hosted.FIRST_TOOL_ENV, raising=False)
    else:
        monkeypatch.setenv(hosted.FIRST_TOOL_ENV, first_tool)
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return REPLY

    hosted.groq_transport(pacer=hosted.Pacer(10**6), create=create)(request)
    return seen


def test_the_first_call_is_free_by_default_because_forcing_it_was_measured_to_hurt(monkeypatch):
    for later in (FIRST, AFTER_TOOL, SENT_BACK, REQUEST):
        assert "tool_choice" not in _sent(later, monkeypatch)
    assert "free" in hosted.settings()["first_call"]
    assert hosted.client(":memory:").model_digest is None  # the cache keys are the plain ones


def test_forcing_the_first_call_is_an_experiment_that_touches_only_the_first_call(monkeypatch):
    assert _sent(FIRST, monkeypatch, "required")["tool_choice"] == "required"
    named = _sent(FIRST, monkeypatch, "named")["tool_choice"]
    assert named["function"]["name"] == "search_examples"
    for later in (AFTER_TOOL, SENT_BACK, REQUEST):  # after a tool, after a send-back, no tools
        assert "tool_choice" not in _sent(later, monkeypatch, "required")
    assert "tool_choice" not in _sent(FIRST, monkeypatch, "0")


def test_a_forced_first_call_is_recorded_and_never_shares_cached_answers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv(hosted.FIRST_TOOL_ENV, "required")
    assert "tool_choice required" in hosted.settings()["first_call"]
    assert hosted.client(":memory:").model_digest == f"{hosted.FIRST_TOOL_KEY}:required"
    monkeypatch.setenv(hosted.FIRST_TOOL_ENV, "named")
    assert hosted.client(":memory:").model_digest == f"{hosted.FIRST_TOOL_KEY}:named"


SYSTEM_REQUEST = {
    **FIRST,
    "messages": [
        {"role": "system", "content": "You classify."},
        {"role": "user", "content": "my message"},
    ],
}


def _messages_sent(request, monkeypatch, search_first=None):
    monkeypatch.delenv("GROQ_REASONING", raising=False)
    monkeypatch.delenv(hosted.FIRST_TOOL_ENV, raising=False)
    if search_first is None:
        monkeypatch.delenv(hosted.SEARCH_FIRST_ENV, raising=False)
    else:
        monkeypatch.setenv(hosted.SEARCH_FIRST_ENV, search_first)
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return REPLY

    hosted.groq_transport(pacer=hosted.Pacer(10**6), create=create)(request)
    return seen["messages"]


def test_the_search_first_line_is_off_by_default_and_changes_nothing(monkeypatch):
    assert _messages_sent(SYSTEM_REQUEST, monkeypatch) == SYSTEM_REQUEST["messages"]
    assert hosted.settings()["search_first"] == "off"
    assert hosted.client(":memory:").model_digest is None


def test_the_search_first_line_is_added_to_the_first_call_only(monkeypatch):
    sent = _messages_sent(SYSTEM_REQUEST, monkeypatch, "1")
    assert sent[0]["content"] == "You classify." + hosted.SEARCH_FIRST_TEXT
    assert sent[1] == SYSTEM_REQUEST["messages"][1]  # the customer's message is untouched
    assert SYSTEM_REQUEST["messages"][0]["content"] == "You classify."  # the request was copied
    later = {
        **SYSTEM_REQUEST,
        "messages": [*SYSTEM_REQUEST["messages"], *AFTER_TOOL["messages"][1:]],
    }
    assert _messages_sent(later, monkeypatch, "1") == later["messages"]  # no line after a tool
    assert _messages_sent(REQUEST, monkeypatch, "1") == REQUEST["messages"]  # no tools, no line
    assert "always call search_examples" in hosted.settings()["search_first"]


def test_the_search_first_line_and_a_repeat_tag_never_share_cached_answers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv(hosted.FIRST_TOOL_ENV, raising=False)
    monkeypatch.setenv(hosted.SEARCH_FIRST_ENV, "1")
    assert hosted.client(":memory:").model_digest == "search-first"
    monkeypatch.setenv(hosted.CACHE_TAG_ENV, "r2")
    assert hosted.client(":memory:").model_digest == "search-first|tag-r2"


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


def test_pacing_learns_that_requests_use_less_than_guessed(monkeypatch):
    """The size guess reserves a full reply; once real counts come back, the pacer stops waiting for
    room that was never used (and a request that proves bigger raises the guess again)."""
    monkeypatch.setenv("GROQ_API_KEY", "k")
    seen = []

    class Spy(hosted.Pacer):
        def wait(self, tokens):
            seen.append(tokens)
            return 0.0

    def create(**_):
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 300}}

    call = hosted.groq_transport(pacer=Spy(8000), create=create)
    request = {
        "model": "m",
        "messages": [{"role": "user", "content": "x" * 1000}],
        "max_tokens": 600,
    }
    for _ in range(6):
        call(request)
    assert seen[0] > 800  # the first guess is the full estimate (about 890)
    assert seen[-1] < seen[0] * 0.75  # then it follows what was really used
    assert seen[-1] >= seen[0] * 0.5  # never below half of the estimate


def test_proxy_is_off_unless_both_settings_are_given():
    assert hosted.proxy({}) is None
    ok = {"GROQ_PROXY_URL": "https://p.example.dev/v1/", "GROQ_PROXY_TOKEN": " tok "}
    assert hosted.proxy(ok) == ("https://p.example.dev/v1", "tok")
    for bad in (
        {"GROQ_PROXY_URL": "https://p.example.dev/v1"},  # no token
        {"GROQ_PROXY_TOKEN": "tok"},  # no address
        {"GROQ_PROXY_URL": "http://p.example.dev/v1", "GROQ_PROXY_TOKEN": "tok"},  # not https
        {"GROQ_PROXY_URL": "https://p.example.dev", "GROQ_PROXY_TOKEN": "tok"},  # no /v1
    ):
        with pytest.raises(ValueError):
            hosted.proxy(bad)


def test_connection_with_a_proxy_never_needs_or_sends_the_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_PROXY_URL", "https://p.example.dev/v1")
    monkeypatch.setenv("GROQ_PROXY_TOKEN", "tok")
    url, key, headers = hosted.connection()
    assert url == "https://p.example.dev/v1" and key == "via-proxy"
    assert headers == {"x-proxy-token": "tok"}
    monkeypatch.delenv("GROQ_PROXY_URL")
    monkeypatch.delenv("GROQ_PROXY_TOKEN")
    monkeypatch.setenv("GROQ_API_KEY", "real")
    assert hosted.connection() == (hosted.GROQ_URL, "real", {})


def test_the_token_and_address_are_not_recorded_in_the_plan(monkeypatch):
    monkeypatch.setenv("GROQ_PROXY_URL", "https://secret-name.example.dev/v1")
    monkeypatch.setenv("GROQ_PROXY_TOKEN", "super-secret-token")
    text = repr(hosted.settings()) + hosted.label()
    assert "super-secret-token" not in text and "secret-name" not in text
    assert hosted.settings()["via_proxy"] is True


def test_a_proxys_own_error_body_is_not_a_refused_tool_call():
    class Err(Exception):
        body = {"error": "model not allowed"}  # the proxy's errors carry a plain string

    assert hosted.refused_tool_call(Err(), {}) is None


def test_refused_credentials_stop_the_run_in_plain_words():
    import httpx
    from openai import AuthenticationError

    def deny(**_):
        resp = httpx.Response(401, request=httpx.Request("POST", "https://p.example.dev"))
        raise AuthenticationError("no", response=resp, body={"error": "bad token"})

    call = hosted.groq_transport(create=deny, pacer=hosted.Pacer(10**6))
    with pytest.raises(LLMUnavailable, match="refused our credentials"):
        call({"messages": [{"role": "user", "content": "x"}]})
