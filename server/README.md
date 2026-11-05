# Agent Verifier local server

FastAPI app that serves the page and its live data. A separate uv project, so the pipeline's frozen
`pyproject.toml` and `uv.lock` stay untouched. Binds to 127.0.0.1 only. It never imports
`agent_committee.seal` or the runner and never reads the sealed test (a test enforces the imports).

```
npm --prefix frontend run build:server          # page built with the "api" provider -> frontend/dist-server
uv run --project server python -m verifier_server   # http://127.0.0.1:8000
uv run --project server pytest server/tests         # tests
```

Endpoints: `GET /api/health`, `GET /api/story`, `GET /api/status`, `POST /api/jobs/rehearsal`
(needs Ollama and the header `x-verifier-client: 1`; DEV only, one at a time), `GET /api/jobs/{id}`,
docs at `/api/docs`. The sealed run is not offered as a job: it runs once, by hand.
