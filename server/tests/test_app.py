import ast
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_committee.schemas import AuditBundle, EvidenceLevel, Verdict, VerdictJSON
from verifier_server.app import create_app
from verifier_server.jobs import JobManager, pid_alive
from verifier_server.status import Paths, run_progress
from verifier_server.story import StoryService

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "server" / "src" / "verifier_server"
PREREG_SHA = hashlib.sha256((REPO / "prereg.yaml").read_bytes()).hexdigest()


def verdict(prereg_sha256: str = PREREG_SHA, **kw) -> VerdictJSON:
    return VerdictJSON(
        verdict=Verdict.INVALID,
        evidence_level=EvidenceLevel.REAL,
        audit=AuditBundle(
            git_commit="c" * 40,
            prereg_sha256=prereg_sha256,
            dev_report_sha256="d",
            test_sha256="t",
            model_digest="m",
            layer_fingerprint="f",
        ),
        **{"reasons": ["r"], "accuracy": {}, **kw},
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    shutil.copy(REPO / "prereg.yaml", tmp_path / "prereg.yaml")
    (tmp_path / "data").mkdir()
    shutil.copy(REPO / "data" / "split_manifest.json", tmp_path / "data" / "split_manifest.json")
    (tmp_path / "results" / "rehearsal").mkdir(parents=True)
    return tmp_path


def make_app(
    root: Path,
    ollama: bool = True,
    cmd: list[str] | None = None,
    jobs: JobManager | None = None,
    testing: bool = True,
):
    story = StoryService(Paths(root), with_case=False, script_root=REPO)
    return create_app(
        root,
        ollama=lambda: ollama,
        story=story,
        jobs=jobs or JobManager(root, cmd=cmd),
        serve_page=False,
        testing=testing,
    )


def client(root: Path, ollama: bool = True, cmd: list[str] | None = None) -> TestClient:
    return TestClient(make_app(root, ollama, cmd))


def wait_done(c: TestClient, job_id: str) -> dict:
    for _ in range(200):
        done = c.get(f"/api/jobs/{job_id}").json()
        if done["state"] != "running":
            return done
        time.sleep(0.05)
    raise AssertionError("job did not finish")


H = {"x-verifier-client": "1"}


def test_health(root):
    assert client(root).get("/api/health").json() == {"status": "ok"}


def test_story_is_404_without_a_verdict(root):
    assert client(root).get("/api/story").status_code == 404


def test_story_prefers_the_sealed_verdict_and_refreshes_when_it_changes(root):
    c = client(root)
    paths = Paths(root)
    paths.rehearsal_verdict.write_text(verdict(rehearsal=True).model_dump_json(), encoding="utf-8")
    body = c.get("/api/story").json()
    assert body["source"] == "rehearsal"
    assert body["case"] is None
    assert body["rules"]["price_tables"] == len(body["price_tables"])
    time.sleep(0.05)
    paths.sealed_verdict.write_text(verdict().model_dump_json(), encoding="utf-8")
    assert c.get("/api/story").json()["source"] == "sealed"


def test_status_reports_the_sealed_run_state_and_counts(root):
    c = client(root, ollama=False)
    s = c.get("/api/status").json()
    assert s["ollama"] is False
    assert s["sealed_run"]["state"] == "not_started"
    assert s["sealed_run"]["test_cases"] == 3271
    assert s["job"] is None
    (root / "results" / "sealed_run.started").write_text("abc\n", encoding="utf-8")
    s = c.get("/api/status").json()["sealed_run"]
    assert s["state"] == "started_not_finished"
    assert s["started_at_commit"] == "abc"


def test_run_progress_counts_only_run_phase_rows(tmp_path):
    db = tmp_path / "llm.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cache (key TEXT, phase TEXT, arm TEXT)")
    con.executemany(
        "INSERT INTO cache VALUES (?, ?, ?)",
        [
            ("a", "run", "prompt"),
            ("b", "run", "prompt"),
            ("c", "build", "prompt"),
            ("d", "run", "agent"),
        ],
    )
    con.commit()
    con.close()
    assert run_progress(db) == ({"prompt": 2, "agent": 1}, None)
    assert run_progress(tmp_path / "missing.sqlite") == ({}, None)


def test_rehearsal_needs_the_client_header(root):
    assert client(root).post("/api/jobs/rehearsal").status_code == 403


def test_rehearsal_stops_when_ollama_is_down(root):
    r = client(root, ollama=False).post("/api/jobs/rehearsal", headers={"x-verifier-client": "1"})
    assert r.status_code == 503


def test_rehearsal_job_runs_once_at_a_time_and_reports_its_output(root):
    cmd = [sys.executable, "-c", "import time; print('hello'); time.sleep(0.6)"]
    c = client(root, cmd=cmd)
    h = {"x-verifier-client": "1"}
    job = c.post("/api/jobs/rehearsal", headers=h).json()
    assert job["state"] == "running"
    assert c.post("/api/jobs/rehearsal", headers=h).status_code == 409
    assert c.get("/api/status").json()["job"]["id"] == job["id"]
    for _ in range(100):
        done = c.get(f"/api/jobs/{job['id']}").json()
        if done["state"] != "running":
            break
        time.sleep(0.05)
    assert done["state"] == "succeeded"
    assert done["output_tail"] == ["hello"]
    assert c.get("/api/jobs/nope").status_code == 404


def test_other_host_names_are_refused(root):
    r = client(root).get("/api/health", headers={"host": "evil.example"})
    assert r.status_code == 400


def test_the_server_never_imports_the_seal_or_the_runner():
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for n in names:
                assert "seal" not in n.split(".") and "runner" not in n.split("."), (path, n)


RUNTIME_IMPORT_CHECK = """
import ast, importlib, sys
from pathlib import Path
from fastapi.testclient import TestClient
from verifier_server.app import create_app
from verifier_server.status import Paths
from verifier_server.story import StoryService
root, repo = Path(sys.argv[1]), Path(sys.argv[2])
story = StoryService(Paths(root), with_case=False, script_root=repo)
c = TestClient(create_app(root, ollama=lambda: False, story=story, serve_page=False, testing=True))
assert c.get("/api/story").status_code == 200
assert c.get("/api/status").status_code == 200
script = repo / "scripts" / "export_frontend_data.py"
for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
    if isinstance(node, ast.ImportFrom) and node.module:
        importlib.import_module(node.module)
    elif isinstance(node, ast.Import):
        for a in node.names:
            importlib.import_module(a.name)
bad = [m for m in sys.modules
       if m.startswith("agent_committee") and m.split(".")[-1] in ("seal", "runner")]
assert not bad, bad
print("clean")
"""


def test_s10_serving_the_story_loads_neither_the_seal_nor_the_runner(root):
    """Runtime check in a fresh process: covers the exec'd export script and every transitive
    import (including the ones replay_case does inside the function), unlike the AST scan."""
    Paths(root).rehearsal_verdict.write_text(
        verdict(rehearsal=True).model_dump_json(), encoding="utf-8"
    )
    done = subprocess.run(
        [sys.executable, "-c", RUNTIME_IMPORT_CHECK, str(root), str(REPO)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "clean"


def test_s2_a_live_rehearsal_from_an_earlier_server_blocks_a_new_one(root):
    jobs = JobManager(root, cmd=[sys.executable, "-c", "pass"])
    jobs.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")  # this process: alive
    r = TestClient(make_app(root, jobs=jobs)).post("/api/jobs/rehearsal", headers=H)
    assert r.status_code == 409
    assert str(os.getpid()) in r.json()["detail"]


def test_s2_a_dead_or_stale_pid_file_does_not_block(root):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    jobs = JobManager(root, cmd=[sys.executable, "-c", "pass"])
    jobs.pid_file.write_text(f"{dead.pid}\n", encoding="utf-8")
    assert jobs.orphan_pid() is None
    jobs.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    old = time.time() - jobs.timeout_s - 60
    os.utime(jobs.pid_file, (old, old))  # older than the timeout: the pid may have been reused
    assert jobs.orphan_pid() is None
    c = TestClient(make_app(root, jobs=jobs))
    job = c.post("/api/jobs/rehearsal", headers=H).json()
    assert wait_done(c, job["id"])["state"] == "succeeded"
    assert not jobs.pid_file.exists()  # removed when the job ends


def wait_pid(jobs: JobManager) -> int:
    for _ in range(200):
        if (pid := jobs.orphan_pid()) is not None:
            return pid
        time.sleep(0.02)
    raise AssertionError("no pid file")


def test_s2_the_job_is_killed_after_the_timeout(root):
    jobs = JobManager(
        root, cmd=[sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=1.0
    )
    c = TestClient(make_app(root, jobs=jobs))
    job = c.post("/api/jobs/rehearsal", headers=H).json()
    pid = wait_pid(jobs)
    done = wait_done(c, job["id"])
    assert done["state"] == "failed"
    assert "timeout" in done["output_tail"][-1]
    assert not jobs.pid_file.exists()
    assert not pid_alive(pid)


def test_s2_server_shutdown_kills_the_rehearsal_child(root):
    jobs = JobManager(root, cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
    with TestClient(make_app(root, jobs=jobs)) as c:  # the context runs the lifespan
        job = c.post("/api/jobs/rehearsal", headers=H).json()
        pid = wait_pid(jobs)
    for _ in range(100):
        if jobs.get(job["id"]).state != "running":
            break
        time.sleep(0.05)
    assert jobs.get(job["id"]).state == "failed"
    assert not pid_alive(pid)


def test_s4_a_verdict_caught_mid_write_is_503_not_500(root):
    Paths(root).rehearsal_verdict.write_text('{"verdict": "INVA', encoding="utf-8")
    c = TestClient(make_app(root), raise_server_exceptions=False)
    assert c.get("/api/story").status_code == 503


def test_s5_a_broken_manifest_does_not_break_the_status(root):
    (root / "data" / "split_manifest.json").write_text("{", encoding="utf-8")
    r = client(root).get("/api/status")
    assert r.status_code == 200
    assert r.json()["sealed_run"]["test_cases"] is None


def test_s7_any_start_error_ends_the_job_so_the_next_one_can_start(root):
    c = client(root, cmd=[sys.executable, "-c", "pass", "bad\0arg"])  # Popen raises ValueError
    job = c.post("/api/jobs/rehearsal", headers=H).json()
    done = wait_done(c, job["id"])
    assert done["state"] == "failed"
    assert "ValueError" in done["output_tail"][-1]
    assert c.post("/api/jobs/rehearsal", headers=H).status_code == 202


def test_s8_a_locked_cache_is_reported_not_shown_as_zero(root):
    db = Paths(root).cache
    db.parent.mkdir()
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cache (key TEXT, phase TEXT, arm TEXT)")
    con.execute("INSERT INTO cache VALUES ('a', 'run', 'prompt')")
    con.commit()
    con.execute("BEGIN EXCLUSIVE")  # a writer holds the lock
    try:
        s = client(root).get("/api/status").json()["sealed_run"]
        assert s["answers_cached"] == {}
        assert "locked" in s["answers_cached_error"]
    finally:
        con.rollback()
        con.close()
    s = client(root).get("/api/status").json()["sealed_run"]
    assert s["answers_cached"] == {"prompt": 1}
    assert s["answers_cached_error"] is None


def write_rehearsal(root: Path, **kw) -> None:
    Paths(root).rehearsal_verdict.write_text(
        verdict(rehearsal=True, **kw).model_dump_json(), encoding="utf-8"
    )


def case_story(root: Path, replay) -> StoryService:
    """A StoryService with the case on, its replay swapped for `replay(root)`."""
    story = StoryService(Paths(root), with_case=True, script_root=REPO)
    story._export.replay_case = replay
    return story


def repo_cache_state() -> list:
    out = []
    for name in ("llm.sqlite", "llm.sqlite-journal", "llm.sqlite-wal"):
        p = REPO / "cache" / name
        out.append((name, p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (name,))
    return out


def test_fixture_root_story_never_touches_the_real_repo_cache(root):
    """The real export code with the case on: it gets the fixture root, finds no cache there,
    returns no case, and the repo's cache/ files are byte-for-byte untouched."""
    write_rehearsal(root)
    seen: list[Path] = []
    story = StoryService(Paths(root), with_case=True, script_root=REPO)
    real = story._export.replay_case

    def spy(r):
        seen.append(Path(r))
        return real(r)

    story._export.replay_case = spy
    before = repo_cache_state()
    c = TestClient(create_app(root, story=story, serve_page=False, testing=True))
    body = c.get("/api/story").json()
    assert seen == [root]
    assert body["case"] is None
    assert not (root / "cache").exists()  # nothing created in the fixture either
    assert repo_cache_state() == before


def test_case_refreshes_after_a_wal_commit_with_the_writer_still_open(root):
    """In WAL mode a commit lands in llm.sqlite-wal and the main file is untouched until a
    checkpoint. The story must still pick up the newly committed data on the next request."""
    import sqlite3

    write_rehearsal(root)
    cache = Paths(root).cache
    cache.parent.mkdir()
    writer = sqlite3.connect(cache, isolation_level=None)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")  # keep every commit in the -wal file
        writer.execute("CREATE TABLE t (n INTEGER)")
        writer.execute("INSERT INTO t VALUES (1)")

        def replay(r):
            con = sqlite3.connect(f"file:{Path(r) / 'cache' / 'llm.sqlite'}?mode=ro", uri=True)
            try:
                return {"n": con.execute("SELECT max(n) FROM t").fetchone()[0]}
            finally:
                con.close()

        story = case_story(root, replay)
        assert story.get()["case"] == {"n": 1}
        main_before = cache.stat().st_mtime_ns, cache.stat().st_size
        writer.execute("INSERT INTO t VALUES (2)")  # committed, writer stays open
        assert (cache.stat().st_mtime_ns, cache.stat().st_size) == main_before
        assert story.get()["case"] == {"n": 2}
    finally:
        writer.close()


def test_a_prereg_that_does_not_match_the_verdict_is_refused(root):
    write_rehearsal(root, prereg_sha256="0" * 64)
    r = TestClient(make_app(root)).get("/api/story")
    assert r.status_code == 503
    assert "sha256" in r.json()["detail"]


def test_a_rehearsal_at_the_sealed_path_is_refused(root):
    Paths(root).sealed_verdict.write_text(
        verdict(rehearsal=True).model_dump_json(), encoding="utf-8"
    )
    assert TestClient(make_app(root)).get("/api/story").status_code == 503


def test_r4_s1_editing_only_the_prereg_invalidates_a_warm_story(root):
    """Codex R4-S1: a warm service must not keep serving a story whose prereg no longer matches
    the verdict; it must answer exactly as a fresh service does."""
    write_rehearsal(root)
    warm = TestClient(make_app(root))
    assert warm.get("/api/story").status_code == 200  # built and cached
    prereg = Paths(root).prereg
    prereg.write_bytes(prereg.read_bytes() + b"\n# edited\n")  # verdict untouched
    fresh = TestClient(make_app(root)).get("/api/story")
    assert fresh.status_code == 503
    again = warm.get("/api/story")
    assert again.status_code == 503
    assert again.json() == fresh.json()
    shutil.copy(REPO / "prereg.yaml", prereg)  # the matching bytes back: served again
    assert warm.get("/api/story").status_code == 200


def test_a_verdict_edit_with_the_same_mtime_still_rebuilds(root):
    write_rehearsal(root)
    story = StoryService(Paths(root), with_case=False, script_root=REPO)
    assert story.get()["verdict"]["reasons"] == ["r"]
    path = Paths(root).rehearsal_verdict
    st = path.stat()
    path.write_text(verdict(rehearsal=True, reasons=["changed"]).model_dump_json(), "utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # same mtime as before
    assert story.get()["verdict"]["reasons"] == ["changed"]


def test_s6_the_replay_runs_outside_the_story_lock_and_only_when_the_cache_changed(root):
    write_rehearsal(root)
    calls: list[bool] = []
    holder: dict = {}

    def replay(r):
        calls.append(holder["story"]._lock.locked())
        return {"id": len(calls)}

    story = holder["story"] = case_story(root, replay)
    c = TestClient(create_app(root, story=story, serve_page=False, testing=True))
    assert c.get("/api/story").json()["case"] == {"id": 1}
    assert calls == [False]  # the story lock was free during the replay
    write_rehearsal(root, reasons=["new"])  # new verdict, same (absent) cache: no second replay
    body = c.get("/api/story").json()
    assert body["verdict"]["reasons"] == ["new"]
    assert body["case"] == {"id": 1}
    assert len(calls) == 1
    cache = Paths(root).cache
    cache.parent.mkdir()
    cache.write_bytes(b"")  # the cache changed: replay again on the next rebuild
    write_rehearsal(root, reasons=["newer"])
    assert c.get("/api/story").json()["case"] == {"id": 2}


def test_s6_a_slow_replay_does_not_block_a_built_story(root):
    write_rehearsal(root)
    started, release = threading.Event(), threading.Event()
    replays: list[int] = []

    def replay(r):
        replays.append(1)
        if len(replays) > 1:
            started.set()
            release.wait(10)
        return None

    story = case_story(root, replay)
    assert story.get()["source"] == "rehearsal"
    cache = Paths(root).cache
    cache.parent.mkdir()
    cache.write_bytes(b"")
    Paths(root).sealed_verdict.write_text(verdict().model_dump_json(), encoding="utf-8")
    t = threading.Thread(target=story.get)
    t.start()  # rebuild for the sealed verdict, stuck in the replay
    assert started.wait(10)
    assert not story._lock.locked()  # other readers are not held behind the replay
    release.set()
    t.join(10)
    assert story.get()["source"] == "sealed"


def test_s9_the_test_client_host_is_refused_outside_tests(root):
    prod = TestClient(make_app(root, testing=False))
    assert prod.get("/api/health").status_code == 400  # Host: testserver
    assert prod.get("/api/health", headers={"host": "127.0.0.1:8000"}).status_code == 200
    assert client(root).get("/api/health").status_code == 200
