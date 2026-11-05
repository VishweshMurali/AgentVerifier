import ast
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_committee.schemas import AuditBundle, EvidenceLevel, Verdict, VerdictJSON
from verifier_server.app import create_app
from verifier_server.jobs import JobManager
from verifier_server.status import Paths, run_progress
from verifier_server.story import StoryService

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "server" / "src" / "verifier_server"


def verdict(**kw) -> VerdictJSON:
    return VerdictJSON(
        verdict=Verdict.INVALID,
        evidence_level=EvidenceLevel.REAL,
        audit=AuditBundle(
            git_commit="c" * 40,
            prereg_sha256="p",
            dev_report_sha256="d",
            test_sha256="t",
            model_digest="m",
            layer_fingerprint="f",
        ),
        reasons=["r"],
        accuracy={},
        **kw,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    shutil.copy(REPO / "prereg.yaml", tmp_path / "prereg.yaml")
    (tmp_path / "data").mkdir()
    shutil.copy(REPO / "data" / "split_manifest.json", tmp_path / "data" / "split_manifest.json")
    (tmp_path / "results" / "rehearsal").mkdir(parents=True)
    return tmp_path


def client(root: Path, ollama: bool = True, cmd: list[str] | None = None) -> TestClient:
    story = StoryService(Paths(root), with_case=False, script_root=REPO)
    app = create_app(
        root,
        ollama=lambda: ollama,
        story=story,
        jobs=JobManager(root, cmd=cmd),
        serve_page=False,
    )
    return TestClient(app)


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
    assert run_progress(db) == {"prompt": 2, "agent": 1}
    assert run_progress(tmp_path / "missing.sqlite") == {}


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
