"""Public mode (settings.py): the hosted agent, its limits, per-visitor uploads, PII masking."""

import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_app import H, wait_done
from test_app import root as _root_fixture

from verifier_server import flow, guide, settings
from verifier_server.app import OWN_HEADER, create_app
from verifier_server.status import Paths, tokens_today
from verifier_server.story import StoryService

REPO = Path(__file__).resolve().parents[2]
root = _root_fixture
PUBLIC_ENV = {"VERIFIER_PUBLIC": "1", "AGENT_PROVIDER": "groq", "GROQ_API_KEY": "test-key"}
DATA = {
    "datasets": [
        {"option": "banking77", "status": "built in"},
        {"option": "mine_abc", "status": "split built", "own": True},
    ],
    "labels": {"banking77": ["a", "b", "c"], "mine_abc": ["x", "y"]},
    "plan_sizes": {"0.10": 168, "0.20": 40},
    "eval_sizes": {"banking77": 900, "mine_abc": 90},
    "speeds": {"banking77": {}, "mine_abc": {}},
    "seconds_per_case": {"agent": 1.05},
    "price_tables": [],
    "saved_run": None,
}
OK = {"scenario": "banking77", "gap": 0.2, "arms": ["agent", "linear"], "scope": ["a"]}


def public_app(root: Path, ollama: bool = False, **env: str):
    cfg = settings.load({**PUBLIC_ENV, **env})
    story = StoryService(Paths(root), with_case=False, script_root=REPO)
    return create_app(
        root, ollama=lambda: ollama, story=story, serve_page=False, testing=True, settings=cfg
    )


def test_local_is_the_default():
    cfg = settings.load({})
    assert not cfg.public and cfg.provider is None
    assert cfg.hosts == settings.LOCAL_HOSTS and cfg.arms is None and cfg.daily_tokens is None
    assert cfg.view()["agent"] == {
        "provider": "ollama",
        "name": "Local Ollama agent",
        "model": "llama3.1:8b via Ollama",
        "pinned": True,
    }


def test_public_needs_a_hosted_agent_and_its_key():
    with pytest.raises(ValueError, match="hosted agent"):
        settings.load({"VERIFIER_PUBLIC": "1"})
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        settings.load({"AGENT_PROVIDER": "groq"})
    with pytest.raises(ValueError, match="AGENT_PROVIDER"):
        settings.load({"AGENT_PROVIDER": "other"})


def test_public_settings():
    cfg = settings.load(PUBLIC_ENV)
    assert cfg.public and cfg.provider == "groq" and cfg.hosts == ("*",)
    assert cfg.arms == settings.PUBLIC_ARMS and cfg.gaps == settings.PUBLIC_GAPS
    assert "random_forest" not in cfg.arms and "boosting" not in cfg.arms
    agent = cfg.view()["agent"]
    assert agent["pinned"] is False and "not pinned" in agent["model"] and "Groq" in agent["model"]


def test_hosts_named_by_the_host_are_the_only_ones_answered():
    cfg = settings.load({**PUBLIC_ENV, "VERIFIER_ALLOWED_HOSTS": "demo.onrender.com"})
    assert cfg.hosts == ("127.0.0.1", "localhost", "demo.onrender.com")
    cfg = settings.load({"VERIFIER_ALLOWED_HOSTS": "a.example, b.example"})
    assert cfg.hosts == ("127.0.0.1", "localhost", "a.example", "b.example")


def test_a_hosted_name_is_answered_only_when_allowed(root):
    anyone = TestClient(public_app(root))  # a host's proxy and health check pick the name
    assert anyone.get("/api/health", headers={"host": "10.0.0.7:10000"}).status_code == 200
    named = TestClient(public_app(root, VERIFIER_ALLOWED_HOSTS="demo.onrender.com"))
    assert named.get("/api/health", headers={"host": "demo.onrender.com"}).status_code == 200
    assert named.get("/api/health", headers={"host": "evil.example"}).status_code == 400
    local = TestClient(create_app(root, serve_page=False, testing=True))
    assert local.get("/api/health", headers={"host": "demo.onrender.com"}).status_code == 400


def test_status_names_the_hosted_agent_and_the_budget(root):
    status = TestClient(public_app(root)).get("/api/status").json()
    assert status["public"] is True
    assert status["agent"]["provider"] == "groq" and status["agent"]["ready"] is True
    assert status["budget"]["daily_tokens"] == settings.DAILY_TOKENS
    assert status["budget"]["used_today"] == 0


def test_rehearsal_and_bring_your_agent_are_off(root, monkeypatch):
    monkeypatch.setattr(flow, "flow_data", lambda _root, _cfg=None: DATA)
    c = TestClient(public_app(root))
    assert c.post("/api/jobs/rehearsal", headers=H).status_code == 403
    body = {"dataset": "banking77", "url": "http://127.0.0.1:9/x"}
    assert c.post("/api/agent/check", json=body, headers=H).status_code == 403
    own = {**OK, "agent": {"url": "http://127.0.0.1:9/x", "usd_per_message": 0.01}}
    assert c.post("/api/jobs/quick", json=own, headers=H).status_code == 422


def test_public_narrows_models_and_gaps():
    cfg = settings.load(PUBLIC_ENV)
    assert flow.quick_command(OK, DATA, cfg)[1:4] == ["-u", "-m", "agent_committee.quick"]
    with pytest.raises(flow.FlowRefused, match="arms"):
        flow.quick_command({**OK, "arms": ["agent", "random_forest"]}, DATA, cfg)
    with pytest.raises(flow.FlowRefused, match="arms"):
        flow.quick_command({**OK, "arms": ["agent", "linear", "prompt"]}, DATA, cfg)
    with pytest.raises(flow.FlowRefused, match="gap"):
        flow.quick_command({**OK, "gap": 0.05}, DATA, cfg)


def test_a_public_test_runs_without_ollama(root, monkeypatch):
    monkeypatch.setattr(flow, "flow_data", lambda _root, _cfg=None: DATA)
    monkeypatch.setattr(
        flow, "quick_command", lambda choice, data, cfg=None: [sys.executable, "-c", "print(1)"]
    )
    c = TestClient(public_app(root, ollama=False))
    r = c.post("/api/jobs/quick", json=OK, headers=H)
    assert r.status_code == 202
    assert wait_done(c, r.json()["id"])["state"] == "succeeded"


def test_status_and_busy_errors_never_show_another_visitors_job(root, monkeypatch):
    monkeypatch.setattr(flow, "flow_data", lambda _root, _cfg=None: DATA)
    slow = [sys.executable, "-c", "import time; print('secret_upload_abc123'); time.sleep(3)"]
    monkeypatch.setattr(flow, "quick_command", lambda choice, data, cfg=None: slow)
    c = TestClient(public_app(root, VERIFIER_QUEUE="1"))
    job = c.post("/api/jobs/quick", json=OK, headers=H).json()
    time.sleep(0.5)
    status = c.get("/api/status").json()
    assert status["job"] == {"kind": "quick", "state": "running"}
    assert job["id"] not in c.get("/api/status").text
    assert "secret_upload_abc123" not in c.get("/api/status").text
    # the next visitor waits in line, and is told only their own place
    waiting = c.post("/api/jobs/quick", json=OK, headers=H)
    assert waiting.status_code == 202
    assert waiting.json()["state"] == "queued" and waiting.json()["position"] == 1
    assert job["id"] not in waiting.text
    # the line holds one: the next is refused, without naming anyone's job
    full = c.post("/api/jobs/quick", json=OK, headers=H)
    assert (
        full.status_code == 409
        and job["id"] not in full.text
        and waiting.json()["id"] not in full.text
    )
    assert "line is full" in full.json()["detail"]
    wait_done(c, job["id"])
    wait_done(c, waiting.json()["id"])


def _spend(cache: Path, tokens: int, ts: float, cached: int = 0) -> None:
    from agent_committee.llm import LLMClient

    cache.parent.mkdir(parents=True, exist_ok=True)
    LLMClient(cache, phase="build", transport=lambda request: {}).close()  # creates the ledger
    con = sqlite3.connect(cache)
    con.execute(
        "INSERT INTO calls (key, phase, arm, status, cached, attempts, wall_s, latency_s, "
        "prompt_tokens, completion_tokens, ts) VALUES ('k','build','agent','ok',?,1,0,0,?,0,?)",
        (cached, tokens, ts),
    )
    con.commit()
    con.close()


def test_tokens_today_counts_only_real_calls_since_midnight(tmp_path):
    cache = tmp_path / "cache" / "llm.sqlite"
    assert tokens_today(cache) == 0
    now = 10 * 86400 + 3600.0  # one hour into a UTC day
    _spend(cache, 500, now - 60)
    _spend(cache, 700, now - 7200)  # yesterday
    _spend(cache, 900, now - 30, cached=1)  # a replayed answer costs nothing
    assert tokens_today(cache, now) == 500


def test_a_test_is_refused_when_the_days_tokens_would_not_cover_it(root, monkeypatch):
    monkeypatch.setattr(flow, "flow_data", lambda _root, _cfg=None: DATA)
    monkeypatch.setattr(
        flow, "quick_command", lambda choice, data, cfg=None: [sys.executable, "-c", "print(1)"]
    )
    # 40 planned cases x 900 tokens = 36,000 needed; only 30,000 of the day's budget is left
    _spend(Paths(root).cache, settings.DAILY_TOKENS - 30_000, time.time())
    c = TestClient(public_app(root))
    r = c.post("/api/jobs/quick", json=OK, headers=H)
    assert r.status_code == 429 and "budget" in r.json()["detail"]


def test_an_upload_is_listed_only_for_the_visitor_who_added_it(root, monkeypatch):
    monkeypatch.setattr(flow, "flow_data", lambda _root, _cfg=None: DATA)
    monkeypatch.setattr(flow, "library", lambda _root, name: [{"text": "t", "label": "x"}])
    c = TestClient(public_app(root))
    seen = c.get("/api/flow").json()
    assert [d["option"] for d in seen["datasets"]] == ["banking77"]
    assert "mine_abc" not in seen["labels"] and "mine_abc" not in seen["eval_sizes"]
    assert c.get("/api/flow/library?dataset=mine_abc").status_code == 404
    mine = {OWN_HEADER: "mine_abc"}
    assert "mine_abc" in c.get("/api/flow", headers=mine).json()["labels"]
    assert c.get("/api/flow/library?dataset=mine_abc", headers=mine).status_code == 200
    stranger = {**OK, "scenario": "mine_abc", "scope": ["x"]}
    assert c.post("/api/jobs/quick", json=stranger, headers=H).status_code == 422


def test_locally_every_upload_is_listed(root, monkeypatch):
    monkeypatch.setattr(flow, "flow_data", lambda _root, _cfg=None: DATA)
    local = TestClient(create_app(root, serve_page=False, testing=True))
    assert "mine_abc" in local.get("/api/flow").json()["labels"]


def test_a_public_upload_gets_a_private_name_and_a_smaller_limit(root, monkeypatch):
    seen = {}

    def add_own(root, name, fmt, raw, phrase, title, max_bytes, generic=False):
        seen.update(name=name, max_bytes=max_bytes)
        return {"name": name}

    monkeypatch.setattr(flow, "flow_data", lambda _root, _cfg=None: DATA)
    monkeypatch.setattr(flow, "add_own", add_own)
    c = TestClient(public_app(root))
    q = "name=mine&format=csv&item=a+message&items=messages&label_kind=topic"
    r = c.post(f"/api/own?{q}", content=b"text,label\n", headers=H)
    assert r.status_code == 201
    assert seen["name"].startswith("mine_") and len(seen["name"]) == len("mine_") + 12
    assert seen["max_bytes"] == settings.PUBLIC_OWN_MAX_BYTES


def test_the_public_guide_never_calls_the_model():
    def call(body):
        raise AssertionError("the model was called")

    payload = {"question": "why?", "facts": "3 of 95 wrong", "fallback": "plain answer"}
    out = guide.guide_reply(payload, call, False)
    assert out == {"reply": "plain answer", "worded_by": "code", "checked": True, "rejected": []}


def _csv(rows):
    return ("text,label\n" + "".join(f"{t},{label}\n" for t, label in rows)).encode()


def test_an_upload_is_masked_before_ids_and_files_are_made():
    rows = [("write to jo@example.com about my card", "card"), ("hello there", "greet")]
    cases, masked = flow.masked_cases(_csv(rows), "csv")
    assert cases[0].text == "write to [EMAIL] about my card"
    assert masked["kinds"] == {"EMAIL": 1} and masked["messages"] == 1
    # the id is made from the masked text: the same message with another address gets the same id
    other = [("write to al@example.org about my card", "card"), ("hello there", "greet")]
    assert [c.id for c in flow.masked_cases(_csv(other), "csv")[0]] == [c.id for c in cases]


def test_an_id_or_group_holding_a_personal_detail_is_dropped():
    raw = b"id,text,label,group\njo@example.com,hello,greet,+44 20 7946 0958\nr2,bye,leave,g1\n"
    cases, _ = flow.masked_cases(raw, "csv")
    assert "@" not in cases[0].id and cases[0].group is None
    assert cases[1].id == "r2" and cases[1].group == "g1"


def test_nothing_personal_reaches_the_disk(tmp_path):
    shutil.copy(REPO / "prereg.yaml", tmp_path / "prereg.yaml")
    rows = [
        (f"{w} request {i} from user{i}@example.com", w)
        for w in ("refund", "card", "transfer")
        for i in range(120)
    ]
    out = flow.add_own(tmp_path, "mine", "csv", _csv(rows), {
        "item": "a message", "items": "messages", "label_kind": "topic"
    })  # fmt: skip
    assert out["masked"]["kinds"] == {"EMAIL": 360} and out["masked"]["messages"] == 360
    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.name != "prereg.yaml"]
    assert written
    for path in written:
        assert b"@example.com" not in path.read_bytes(), path


def test_the_upload_limit_is_the_deployments(tmp_path):
    shutil.copy(REPO / "prereg.yaml", tmp_path / "prereg.yaml")
    phrase = {"item": "a message", "items": "messages", "label_kind": "topic"}
    with pytest.raises(flow.FlowRefused, match="larger than 1 MB"):
        flow.add_own(tmp_path, "mine", "csv", b"x" * (1024 * 1024 + 1), phrase, "", 1024 * 1024)


def test_old_public_uploads_are_deleted_and_nothing_else(tmp_path):
    """The reader's data is not kept: an upload past its time goes, with its scenario and results;
    a newer one, a running one and every other dataset stay."""
    old, new, busy = "mine_" + "a" * 12, "mine_" + "b" * 12, "mine_" + "c" * 12
    (tmp_path / "sources" / "own").mkdir(parents=True)
    for n in (old, new, busy, "banking77"):
        (tmp_path / "sources" / "own" / f"{n}.jsonl").write_text("{}\n")
        for top in ("scenarios", "results", "data", "sealed"):
            (tmp_path / top / n).mkdir(parents=True)
    now = time.time()
    import os

    for n, age in ((old, 9000), (busy, 9000), (new, 10), ("banking77", 99999)):
        os.utime(tmp_path / "sources" / "own" / f"{n}.jsonl", (now - age, now - age))
    gone = flow.purge_own(tmp_path, 7200, keep={busy}, now=now)
    assert gone == [old]
    for top in ("scenarios", "results", "data", "sealed"):
        assert not (tmp_path / top / old).exists()
    for n in (new, busy, "banking77"):
        assert (tmp_path / "sources" / "own" / f"{n}.jsonl").exists()
        assert (tmp_path / "scenarios" / n).exists()


def test_the_public_site_publishes_no_api_map(root):
    public = TestClient(public_app(root))
    assert public.get("/api/docs").status_code == 404
    assert public.get("/api/openapi.json").status_code == 404
    local = TestClient(create_app(root, serve_page=False, testing=True))
    assert local.get("/api/openapi.json").status_code == 200
