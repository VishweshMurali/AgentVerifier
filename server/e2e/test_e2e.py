"""End to end: the real server process, a fixture data folder and the built page.

Run after `npm --prefix frontend run build:server` (scripts/check.py does this):
    uv run --project server pytest server/e2e

It never touches the real results/ or cache/, never calls a model, and never opens sealed/.
The fixture verdict is the one committed in frontend/public/data/story.json, so the test also
proves the numbers the server hands the page are exactly the pipeline's own.
"""

import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agent_committee.schemas import VerdictJSON

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "frontend" / "dist-server"
STORY = REPO / "frontend" / "public" / "data" / "story.json"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(url: str, method: str = "GET", headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


@pytest.fixture(scope="module")
def committed_story() -> dict:
    return json.loads(STORY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def server(tmp_path_factory, committed_story):
    if not (PAGE / "index.html").exists():
        pytest.fail("frontend/dist-server is missing: run `npm --prefix frontend run build:server`")
    root = tmp_path_factory.mktemp("root")
    shutil.copy(REPO / "prereg.yaml", root / "prereg.yaml")
    (root / "data").mkdir()
    shutil.copy(REPO / "data" / "split_manifest.json", root / "data" / "split_manifest.json")
    (root / "results" / "rehearsal").mkdir(parents=True)
    verdict = VerdictJSON.model_validate(committed_story["verdict"])
    (root / "results" / "rehearsal" / "verdict.json").write_text(
        verdict.model_dump_json(), encoding="utf-8"
    )
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "verifier_server", "--port", str(port), "--root", str(root)]
        + ["--page-dir", str(PAGE), "--no-case"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            pytest.fail(f"server exited early:\n{out}")
        try:
            if call(base + "/api/health")[0] == 200:
                break
        except OSError:
            pass
        time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("server did not answer /api/health within 20 s")
    yield base, root
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_health(server):
    base, _ = server
    assert json.loads(call(base + "/api/health")[1]) == {"status": "ok"}


def test_story_numbers_are_the_pipelines_own(server, committed_story):
    base, _ = server
    status, body = call(base + "/api/story")
    assert status == 200
    story = json.loads(body)
    assert story["source"] == "rehearsal"
    assert story["case"] is None  # --no-case: no LLM cache in this test
    assert story["verdict"] == committed_story["verdict"]
    assert story["price_tables"] == committed_story["price_tables"]
    assert story["rules"] == committed_story["rules"]
    VerdictJSON.model_validate(story["verdict"])


def test_status_reads_the_fixture_not_the_real_results(server):
    base, _ = server
    s = json.loads(call(base + "/api/status")[1])
    assert s["sealed_run"]["state"] == "not_started"
    assert s["sealed_run"]["answers_cached"] == {}
    assert s["rehearsal"]["verdict"] is not None
    assert s["job"] is None


def test_rehearsal_job_is_guarded(server):
    base, _ = server
    assert call(base + "/api/jobs/rehearsal", "POST")[0] == 403
    assert call(base + "/api/jobs/nope")[0] == 404


def test_the_built_page_and_its_assets_are_served(server):
    base, _ = server
    status, body = call(base + "/")
    html = body.decode()
    assert status == 200
    assert '<div id="root">' in html
    assets = re.findall(r'(?:src|href)="\./(assets/[^"]+)"', html)
    assert any(a.endswith(".js") for a in assets) and any(a.endswith(".css") for a in assets)
    for a in assets:
        assert call(f"{base}/{a}")[0] == 200, a
