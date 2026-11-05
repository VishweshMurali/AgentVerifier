"""Agent Verifier local API. Binds to 127.0.0.1 only (see __main__.py).

    GET  /api/health          liveness
    GET  /api/story           the page's data (same shape as frontend/public/data/story.json)
    GET  /api/status          Ollama, sealed-run state and progress, rehearsal
    POST /api/jobs/rehearsal  start a DEV rehearsal (needs Ollama; one at a time)
    GET  /api/jobs/{id}       a job's state and output tail
    /                         the built page (frontend/dist-server), when it exists

Every number comes from the pipeline's own files and deterministic code. Nothing here imports
agent_committee.seal or the runner, and nothing reads the sealed test.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from .jobs import JobBusy, JobManager
from .status import Paths, ollama_up, project_status
from .story import StoryService

ROOT = Path(__file__).resolve().parents[3]
# Writes need this header: a cross-site page cannot send it without a CORS preflight, which fails.
CLIENT_HEADER = "x-verifier-client"


def create_app(
    root: Path = ROOT,
    *,
    ollama: Callable[[], bool] = ollama_up,
    story: StoryService | None = None,
    jobs: JobManager | None = None,
    serve_page: bool = True,
    page_dir: Path | None = None,
    with_case: bool = True,
) -> FastAPI:
    paths = Paths(root)
    # the export code always comes from this checkout; `root` only says where the data lives
    stories = story or StoryService(paths, with_case=with_case, script_root=ROOT)
    manager = jobs or JobManager(root)
    app = FastAPI(title="Agent Verifier", docs_url="/api/docs", openapi_url="/api/openapi.json")
    # refuse other Host names (DNS rebinding); "testserver" is the test client's
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/story")
    def get_story() -> dict[str, Any]:
        try:
            return stories.get()
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        status = project_status(paths, ollama())
        running = manager.running()
        status["job"] = running.view() if running else None
        return status

    @app.post("/api/jobs/rehearsal", status_code=202)
    def start_rehearsal(request: Request) -> dict[str, Any]:
        if request.headers.get(CLIENT_HEADER) != "1":
            raise HTTPException(403, f"send the header {CLIENT_HEADER}: 1")
        if not ollama():
            raise HTTPException(503, "Ollama is not reachable; start it before a rehearsal")
        try:
            return manager.start_rehearsal().view()
        except JobBusy as exc:
            raise HTTPException(409, f"a job is already running: {exc}") from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.view()

    page = page_dir or paths.dist
    if serve_page and page.is_dir():
        app.mount("/", StaticFiles(directory=page, html=True), name="page")
    return app
