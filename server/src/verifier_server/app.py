"""Agent Verifier local API. Binds to 127.0.0.1 only (see __main__.py).

    GET  /api/health          liveness
    GET  /api/story           the page's data (same shape as frontend/public/data/story.json)
    GET  /api/status          Ollama, sealed-run state and progress, rehearsal
    POST /api/jobs/rehearsal  start a DEV rehearsal (needs Ollama; one at a time)
    GET  /api/jobs/{id}       a job's state, output tail and parsed progress
    GET  /api/flow            the test flow's choices (datasets, labels, test sizes, prices)
    GET  /api/flow/library    a fixed DEV sample per label (the page's message graph)
    POST /api/own             add the reader's own .csv/.jsonl: checked, then split once
    POST /api/jobs/quick      start a quick test on DEV (one job at a time; Ollama unless {agent})
    POST /api/agent/check     send 3 practice messages to the user's agent, show the answers
    GET  /api/jobs/{id}/result  a finished quick test's plan, scope and verdict
    /                         the built page (frontend/dist-server), when it exists

Every number comes from the pipeline's own files and deterministic code. Nothing here imports
agent_committee.seal or the runner, and nothing reads the sealed test.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from . import flow, guide
from .jobs import JobBusy, JobManager
from .status import Paths, ollama_up, project_status
from .story import StoryService, VerdictUnreadable

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
    testing: bool = False,
) -> FastAPI:
    paths = Paths(root)
    # the export code always comes from this checkout; `root` only says where the data lives
    stories = story or StoryService(paths, with_case=with_case, script_root=ROOT)
    manager = jobs or JobManager(root)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown()  # a rehearsal never outlives the server

    app = FastAPI(
        title="Agent Verifier",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    # refuse other Host names (DNS rebinding); "testserver" (the test client's) only in tests
    hosts = ["127.0.0.1", "localhost"] + (["testserver"] if testing else [])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/story")
    def get_story() -> dict[str, Any]:
        try:
            return stories.get()
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except VerdictUnreadable as exc:
            raise HTTPException(503, f"the verdict cannot be shown yet: {exc}") from exc

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

    flow_cache: dict[str, Any] = {}

    def flow_choices() -> dict[str, Any]:
        # the menu is cheap; labels and test sizes are kept until the set of ready datasets changes
        from agent_committee import catalog

        # ...or until a file it is built from changes (scenarios, splits, saved models and speeds)
        inputs = [
            *root.glob("scenarios/*/*.yaml"),
            *root.glob("data/*/split_manifest.json"),
            *root.glob("cache/quick-models/summary-*.json"),
        ]
        stamp = max((f.stat().st_mtime_ns for f in inputs), default=0)
        key = repr(
            ([(e["option"], e.get("status")) for e in catalog.menu(root)], len(inputs), stamp)
        )
        if flow_cache.get("key") != key:
            flow_cache.update(key=key, data=flow.flow_data(root))
        return flow_cache["data"]

    @app.get("/api/flow")
    def get_flow() -> dict[str, Any]:
        return flow_choices()

    library_cache: dict[str, list[dict[str, str]]] = {}

    @app.post("/api/own", status_code=201)
    async def add_own(
        request: Request,
        name: str,
        format: str,
        item: str,
        items: str,
        label_kind: str,
        title: str = "",
    ) -> dict[str, Any]:
        """The reader's own data: the raw .csv/.jsonl as the request body (no form parsing)."""
        if request.headers.get(CLIENT_HEADER) != "1":
            raise HTTPException(403, f"send the header {CLIENT_HEADER}: 1")
        raw = await request.body()
        phrase = {"item": item, "items": items, "label_kind": label_kind}
        try:
            out = flow.add_own(root, name, format, raw, phrase, title)
        except flow.FlowRefused as exc:
            raise HTTPException(422, str(exc)) from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(422, f"the file could not be read: {exc}") from exc
        flow_cache.clear()
        return out

    @app.get("/api/flow/library")
    def get_library(dataset: str) -> list[dict[str, str]]:
        if dataset not in flow_choices()["labels"]:
            raise HTTPException(404, f"dataset {dataset!r} is not ready")
        if dataset not in library_cache:
            library_cache[dataset] = flow.library(root, dataset)
        return library_cache[dataset]

    @app.post("/api/jobs/quick", status_code=202)
    async def start_quick(request: Request) -> dict[str, Any]:
        if request.headers.get(CLIENT_HEADER) != "1":
            raise HTTPException(403, f"send the header {CLIENT_HEADER}: 1")
        try:
            choice = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "send the choice as JSON") from exc
        if not isinstance(choice, dict):
            raise HTTPException(400, "send the choice as a JSON object")
        try:
            cmd = flow.quick_command(choice, flow_choices())
        except flow.FlowRefused as exc:
            raise HTTPException(422, str(exc)) from exc
        # the reference agent runs on Ollama; the user's own agent does not need it
        if choice.get("agent") is None and not ollama():
            raise HTTPException(503, "Ollama is not reachable; start it before a test")
        try:
            return manager.start("quick", cmd, flow.QUICK_TIMEOUT_S, flow.read_progress).view()
        except JobBusy as exc:
            raise HTTPException(409, f"a job is already running: {exc}") from exc

    @app.post("/api/agent/check")
    async def check_agent(request: Request) -> dict[str, Any]:
        """Send three practice messages to the user's agent and show what came back."""
        if request.headers.get(CLIENT_HEADER) != "1":
            raise HTTPException(403, f"send the header {CLIENT_HEADER}: 1")
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "send JSON") from exc
        dataset = body.get("dataset") if isinstance(body, dict) else None
        if not isinstance(dataset, str) or dataset not in flow_choices()["labels"]:
            raise HTTPException(422, "send {dataset, url} for a ready dataset")
        from starlette.concurrency import run_in_threadpool

        try:
            return await run_in_threadpool(
                flow.probe_agent, root, body["dataset"], str(body.get("url", ""))
            )
        except flow.FlowRefused as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/guide")
    async def post_guide(request: Request) -> dict[str, Any]:
        """MOCKUP: the guide words a reply from facts the page computed; plain code checks it."""
        if request.headers.get(CLIENT_HEADER) != "1":
            raise HTTPException(403, f"send the header {CLIENT_HEADER}: 1")
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "send JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(400, "send a JSON object")
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(guide.guide_reply, payload)

    @app.get("/api/jobs/{job_id}/result")
    def get_result(job_id: str) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None or job.kind != "quick":
            raise HTTPException(404, "no such quick test")
        if job.state != "succeeded" or "result" not in job.progress:
            raise HTTPException(409, f"the quick test is {job.state}, no result yet")
        try:
            return flow.read_result(root, job.progress["result"])
        except flow.FlowRefused as exc:
            raise HTTPException(500, str(exc)) from exc

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
