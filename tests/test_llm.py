import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from agent_committee.llm import LLMClient, LLMUnavailable, request_key

MSG = [{"role": "user", "content": "hi"}]


def _resp(text="ok", pt=10, ct=2):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
    }


def _conn_error():
    return APIConnectionError(request=httpx.Request("POST", "http://x"))


class Fake:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def client(fake, **kw):
    return LLMClient(":memory:", transport=fake, sleep=lambda s: None, **kw)


def test_cache_hit_skips_transport_and_keeps_tokens():
    fake = Fake([_resp("a", 10, 2)])
    c = client(fake)
    first = c.chat(MSG)
    second = c.chat(MSG)
    assert fake.calls == 1
    assert not first.cached and second.cached
    assert second.content == "a" and second.prompt_tokens == 10
    summary = c.usage_summary()
    assert summary["calls"] == 2 and summary["cache_hits"] == 1


def test_key_depends_on_full_request():
    base = {"model": "m", "messages": MSG, "temperature": 0.0}
    assert request_key(base) != request_key({**base, "temperature": 0.5})
    assert request_key(base) != request_key({**base, "tools": [{"x": 1}]})
    assert request_key(base) == request_key(dict(reversed(list(base.items()))))


def test_retries_then_succeeds():
    fake = Fake([_conn_error(), _conn_error(), _resp()])
    r = client(fake).chat(MSG)
    assert r.attempts == 3 and fake.calls == 3


def test_gives_up_with_llm_unavailable_after_retries():
    fake = Fake([_conn_error()] * 4)
    with pytest.raises(LLMUnavailable):
        client(fake, max_retries=3).chat(MSG)
    assert fake.calls == 4


def test_non_retryable_error_propagates_immediately():
    err = BadRequestError(
        "bad", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None
    )
    fake = Fake([err])
    with pytest.raises(BadRequestError):
        client(fake).chat(MSG)
    assert fake.calls == 1


def test_failure_is_not_cached():
    fake = Fake([_conn_error()] * 4 + [_resp("late")])
    c = client(fake, max_retries=3)
    with pytest.raises(LLMUnavailable):
        c.chat(MSG)
    assert c.chat(MSG).content == "late"


def test_persists_across_instances(tmp_path):
    db = tmp_path / "c.sqlite"
    LLMClient(db, transport=Fake([_resp("x")])).chat(MSG)
    fake2 = Fake([])
    assert LLMClient(db, transport=fake2).chat(MSG).content == "x"
    assert fake2.calls == 0
