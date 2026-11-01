import threading

import httpx
import pytest
from openai import APIConnectionError, BadRequestError, NotFoundError

from agent_committee.llm import LLMClient, LLMUnavailable, LLMUsageMissing, request_key

MSG = [{"role": "user", "content": "hi"}]


def _resp(text="ok", pt=10, ct=2, finish="stop"):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
    }


def _request():
    return httpx.Request("POST", "http://x")


def _conn_error():
    return APIConnectionError(request=_request())


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


def client(fake, phase="build", **kw):
    return LLMClient(":memory:", phase=phase, transport=fake, sleep=lambda s: None, **kw)


def test_cache_hit_skips_transport_and_keeps_tokens():
    fake = Fake([_resp("a", 10, 2)])
    c = client(fake)
    first = c.chat(MSG, arm="prompt")
    second = c.chat(MSG, arm="prompt")
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


def test_arm_is_part_of_the_key_and_tags_the_logs():
    fake = Fake([_resp("a"), _resp("b")])
    c = client(fake)
    assert c.chat(MSG, arm="prompt").content == "a"
    assert c.chat(MSG, arm="agent").content == "b"
    rows = c._db.execute("SELECT DISTINCT arm, phase FROM calls ORDER BY arm").fetchall()
    assert rows == [("agent", "build"), ("prompt", "build")]


def test_phases_do_not_share_cache_rows(tmp_path):
    db = tmp_path / "c.sqlite"
    build = LLMClient(db, phase="build", transport=Fake([_resp("b")]))
    build.chat(MSG, arm="prompt")
    run_fake = Fake([_resp("r")])
    run = LLMClient(db, phase="run", transport=run_fake)
    assert run.chat(MSG, arm="prompt").content == "r"
    assert run_fake.calls == 1
    phases = {p for (p,) in run._db.execute("SELECT phase FROM cache")}
    assert phases == {"build", "run"}


def test_invalid_phase_rejected():
    with pytest.raises(ValueError):
        LLMClient(":memory:", phase="test", transport=Fake([]))  # type: ignore[arg-type]


def test_finish_reason_stored_and_truncation_visible():
    c = client(Fake([_resp("cut", finish="length")]))
    first = c.chat(MSG, arm="prompt", max_tokens=1)
    assert first.finish_reason == "length" and first.truncated
    again = c.chat(MSG, arm="prompt", max_tokens=1)
    assert again.cached and again.truncated


def test_missing_usage_fails_loudly_and_is_not_cached():
    no_usage = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    c = client(Fake([no_usage, _resp("ok")]))
    with pytest.raises(LLMUsageMissing):
        c.chat(MSG, arm="prompt")
    assert c.chat(MSG, arm="prompt").content == "ok"
    assert c.usage_summary()["failed"] == 1


def test_partial_usage_rejected():
    partial = _resp()
    partial["usage"] = {"prompt_tokens": 5}
    with pytest.raises(LLMUsageMissing):
        client(Fake([partial])).chat(MSG, arm="prompt")


def test_retries_then_succeeds():
    fake = Fake([_conn_error(), _conn_error(), _resp()])
    r = client(fake).chat(MSG, arm="prompt")
    assert r.attempts == 3 and fake.calls == 3


def test_gives_up_with_llm_unavailable_and_logs_the_failure():
    fake = Fake([_conn_error()] * 4)
    c = client(fake, max_retries=3)
    with pytest.raises(LLMUnavailable):
        c.chat(MSG, arm="prompt")
    assert fake.calls == 4
    summary = c.usage_summary()
    assert summary["calls"] == 1 and summary["failed"] == 1 and summary["ok"] == 0
    attempts, error = c._db.execute("SELECT attempts, error FROM calls").fetchone()
    assert attempts == 4 and "APIConnectionError" in error


def test_non_retryable_error_propagates_immediately_and_is_logged():
    err = BadRequestError("bad", response=httpx.Response(400, request=_request()), body=None)
    fake = Fake([err])
    c = client(fake)
    with pytest.raises(BadRequestError):
        c.chat(MSG, arm="prompt")
    assert fake.calls == 1 and c.usage_summary()["failed"] == 1


def test_model_not_found_becomes_llm_unavailable():
    err = NotFoundError("no model", response=httpx.Response(404, request=_request()), body=None)
    fake = Fake([err])
    with pytest.raises(LLMUnavailable, match="not found"):
        client(fake).chat(MSG, arm="prompt")
    assert fake.calls == 1


def test_failure_is_not_cached():
    fake = Fake([_conn_error()] * 4 + [_resp("late")])
    c = client(fake, max_retries=3)
    with pytest.raises(LLMUnavailable):
        c.chat(MSG, arm="prompt")
    assert c.chat(MSG, arm="prompt").content == "late"


def test_model_digest_is_part_of_the_key():
    fake = Fake([_resp("a"), _resp("b")])
    a = client(fake, model_digest="sha-1")
    assert a.chat(MSG, arm="prompt").content == "a"
    b = LLMClient(":memory:", phase="build", transport=fake, model_digest="sha-2")
    assert b.chat(MSG, arm="prompt").content == "b"


def test_concurrent_callers_on_one_key_get_one_stored_answer():
    barrier = threading.Barrier(2)
    counter = iter(["first", "second"])

    def transport(request):
        barrier.wait(timeout=5)  # both threads miss the cache before either inserts
        return _resp(next(counter))

    c = LLMClient(":memory:", phase="build", transport=transport)
    out: list[str] = []
    threads = [
        threading.Thread(target=lambda: out.append(c.chat(MSG, arm="prompt").content))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(out)) == 1
    assert c._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 1


def test_persists_across_instances_and_context_manager(tmp_path):
    db = tmp_path / "c.sqlite"
    with LLMClient(db, phase="build", transport=Fake([_resp("x")])) as c1:
        c1.chat(MSG, arm="prompt")
    fake2 = Fake([])
    with LLMClient(db, phase="build", transport=fake2) as c2:
        assert c2.chat(MSG, arm="prompt").content == "x"
    assert fake2.calls == 0
