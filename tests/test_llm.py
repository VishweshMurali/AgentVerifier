import threading

import httpx
import pytest
from openai import APIConnectionError, BadRequestError, NotFoundError

from agent_committee.llm import (
    LLMCacheSchemaError,
    LLMClient,
    LLMUnavailable,
    LLMUsageMissing,
    request_key,
)

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


def test_old_schema_cache_is_refused_with_clear_error(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, request_json TEXT, response_json TEXT)")
    con.commit()
    con.close()
    with pytest.raises(LLMCacheSchemaError, match="phase/arm"):
        LLMClient(db, phase="build", transport=Fake([]))


def test_concurrent_loser_logs_its_own_usage_not_the_winners():
    barrier = threading.Barrier(2)
    usages = iter([(30, 3), (40, 4)])

    def transport(request):
        barrier.wait(timeout=5)
        pt, ct = next(usages)
        return _resp("same", pt, ct)

    c = LLMClient(":memory:", phase="build", transport=transport)
    threads = [threading.Thread(target=lambda: c.chat(MSG, arm="prompt")) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logged = sorted(c._db.execute("SELECT prompt_tokens FROM calls").fetchall())
    assert logged == [(30,), (40,)]


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"choices": []},
        {"choices": [{}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        {"choices": [{"message": "str"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    ],
)
def test_malformed_response_is_a_logged_failure(bad):
    c = client(Fake([bad]))
    with pytest.raises(Exception):  # noqa: B017 - any parse error is acceptable, logging is the point
        c.chat(MSG, arm="prompt")
    assert c.usage_summary()["failed"] == 1
    assert c._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0


def test_persists_across_instances_and_context_manager(tmp_path):
    db = tmp_path / "c.sqlite"
    with LLMClient(db, phase="build", transport=Fake([_resp("x")])) as c1:
        c1.chat(MSG, arm="prompt")
    fake2 = Fake([])
    with LLMClient(db, phase="build", transport=fake2) as c2:
        assert c2.chat(MSG, arm="prompt").content == "x"
    assert fake2.calls == 0


# ---- P1b: SQLite lock fixes (L1-L4), response boundary (F3), served-digest pin (F5) ----------

import sqlite3  # noqa: E402

from agent_committee.llm import (  # noqa: E402
    LLMCacheWriteError,
    MalformedResponse,
    ModelDigestMismatch,
    fetch_served_digest,
)


def file_client(db, fake, **kw):
    return LLMClient(db, phase="build", transport=fake, sleep=lambda s: None, **kw)


def count(db, table, where=""):
    con = sqlite3.connect(db)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0]
    finally:
        con.close()


def test_file_db_uses_wal_and_long_busy_timeout(tmp_path):
    c = file_client(tmp_path / "c.sqlite", Fake([]))
    assert c._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert c._db.execute("PRAGMA busy_timeout").fetchone()[0] >= 30000


def test_open_reader_does_not_block_the_writer(tmp_path):
    db = tmp_path / "c.sqlite"
    c = file_client(db, Fake([_resp("a")]), busy_timeout_s=0.2)
    reader = sqlite3.connect(db, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM cache").fetchone()  # read transaction left open
    try:
        assert c.chat(MSG, arm="prompt").content == "a"
    finally:
        reader.close()
    assert count(db, "cache") == 1 and count(db, "calls", "WHERE status='ok'") == 1


def test_failed_cache_write_is_named_rolled_back_and_recoverable(tmp_path):
    db = tmp_path / "c.sqlite"
    c = file_client(db, Fake([_resp("lost", 11, 3), _resp("later", 5, 1)]), busy_timeout_s=0.1)
    other = sqlite3.connect(db, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")  # another writer holds the lock past the timeout
    with pytest.raises(LLMCacheWriteError) as info:
        c.chat(MSG, arm="prompt")
    assert (info.value.prompt_tokens, info.value.completion_tokens) == (11, 3)
    assert isinstance(info.value.__cause__, sqlite3.OperationalError)
    assert not c._db.in_transaction
    other.execute("ROLLBACK")
    other.close()
    assert count(db, "cache") == 0 and count(db, "calls", "WHERE status='ok'") == 0
    assert c.chat(MSG, arm="prompt").content == "later"  # the client works again


def test_failed_log_write_keeps_the_original_error(tmp_path):
    db = tmp_path / "c.sqlite"
    c = file_client(db, Fake([ValueError("ORIGINAL")]), busy_timeout_s=0.1)
    other = sqlite3.connect(db, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(ValueError, match="ORIGINAL") as info:
            c.chat(MSG, arm="prompt")
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert any("locked" in n for n in info.value.__notes__)
    assert not c._db.in_transaction


def test_cache_row_and_calls_row_are_written_atomically(tmp_path):
    db = tmp_path / "c.sqlite"
    c = file_client(db, Fake([_resp("a", 7, 2)]), busy_timeout_s=0.1)
    other = sqlite3.connect(db, isolation_level=None)
    other.execute(
        "CREATE TRIGGER no_ok BEFORE INSERT ON calls WHEN NEW.status = 'ok' "
        "BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    other.close()
    with pytest.raises(LLMCacheWriteError):
        c.chat(MSG, arm="prompt")
    assert count(db, "cache") == 0  # the cache insert rolled back with the failed calls insert
    assert count(db, "calls", "WHERE status='failed'") == 1


@pytest.mark.parametrize(
    "message",
    [
        None,
        "str",
        {"content": 5},
        {"content": ["x"]},
        {"content": "x", "tool_calls": "oops"},
        {"content": None, "tool_calls": [1]},
        {"content": None, "tool_calls": {"a": 1}},
    ],
)
def test_bad_message_shape_raises_malformed_with_usage_and_ledger(message):
    bad = {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 21, "completion_tokens": 4},
    }
    c = client(Fake([bad]))
    with pytest.raises(MalformedResponse) as info:
        c.chat(MSG, arm="prompt")
    assert (info.value.prompt_tokens, info.value.completion_tokens) == (21, 4)
    assert c._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0
    row = c._db.execute("SELECT status, prompt_tokens, completion_tokens FROM calls").fetchone()
    assert row == ("failed", 21, 4)


@pytest.mark.parametrize("choices", [[], None, "x", ["x"]])
def test_bad_choices_raise_malformed(choices):
    bad = {"choices": choices, "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    with pytest.raises(MalformedResponse) as info:
        client(Fake([bad])).chat(MSG, arm="prompt")
    assert info.value.prompt_tokens == 3


def test_valid_shapes_are_accepted():
    ok = _resp(None)
    ok["choices"][0]["message"]["tool_calls"] = [{"id": "1", "function": {"name": "f"}}]
    assert client(Fake([ok])).chat(MSG, arm="prompt").content == ""
    plain = _resp("hi")
    plain["choices"][0]["message"]["tool_calls"] = None
    assert client(Fake([plain])).chat(MSG, arm="prompt").content == "hi"


def test_cached_bad_reply_behaves_like_a_live_one():
    c = client(Fake([_resp("fine", 9, 2)]))
    c.chat(MSG, arm="prompt")
    c._db.execute("UPDATE cache SET response_json = ?", ('{"content": 5}',))
    with pytest.raises(MalformedResponse) as info:
        c.chat(MSG, arm="prompt")
    assert (info.value.prompt_tokens, info.value.completion_tokens) == (9, 2)
    last = c._db.execute("SELECT status, prompt_tokens FROM calls ORDER BY id DESC").fetchone()
    assert last == ("failed", 9)
    c._db.execute("UPDATE cache SET response_json = ?", ("not json",))
    with pytest.raises(MalformedResponse):
        c.chat(MSG, arm="prompt")


def test_outage_is_still_llm_unavailable_not_malformed():
    with pytest.raises(LLMUnavailable):
        client(Fake([_conn_error()] * 4), max_retries=3).chat(MSG, arm="prompt")


DIGEST = "ab" * 32


def test_matching_served_digest_is_accepted():
    c = client(Fake([_resp()]), model_digest=DIGEST, served_digest=lambda m: "sha256:" + DIGEST)
    assert c.chat(MSG, arm="prompt").content == "ok"


def test_digest_mismatch_refuses_before_touching_the_cache(tmp_path):
    db = tmp_path / "sub" / "c.sqlite"
    with pytest.raises(ModelDigestMismatch) as info:
        file_client(db, Fake([]), model_digest=DIGEST, served_digest=lambda m: "cd" * 32)
    assert (info.value.configured, info.value.served) == (DIGEST, "cd" * 32)
    assert not db.exists()


def test_production_client_resolves_the_digest_from_ollama(monkeypatch):
    import agent_committee.llm as llm

    seen = []
    monkeypatch.setattr(llm, "fetch_served_digest", lambda m: seen.append(m) or "ef" * 32)
    with pytest.raises(ModelDigestMismatch):
        LLMClient(":memory:", phase="build", model_digest=DIGEST)  # no transport: real path
    assert seen == ["llama3.1:8b"]


def _tags_client(payload=None, status=200):
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_served_digest_reads_the_tags_endpoint():
    tags = {"models": [{"name": "other", "digest": "x"}, {"name": "llama3.1:8b", "digest": DIGEST}]}
    assert fetch_served_digest("llama3.1:8b", http_client=_tags_client(tags)) == DIGEST


@pytest.mark.parametrize(
    "payload,status", [({"models": []}, 200), ({"oops": 1}, 200), ({}, 500), ([1], 200)]
)
def test_fetch_served_digest_failures_are_llm_unavailable(payload, status):
    with pytest.raises(LLMUnavailable):
        fetch_served_digest("llama3.1:8b", http_client=_tags_client(payload, status))
