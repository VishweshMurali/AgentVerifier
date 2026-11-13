# Agent Verifier local server

FastAPI app that serves the live page and runs quick tests. A separate uv project, so the
pipeline's `pyproject.toml` and `uv.lock` stay untouched. Binds to 127.0.0.1 only. It never imports
`agent_committee.seal` and never reads the sealed test (a test enforces the imports). Quick tests
run as a child process (`python -m agent_committee.quick`), one at a time.

```
npm --prefix frontend run build:server               # page built with the "api" provider -> frontend/dist-server
uv run --project server python -m verifier_server    # http://127.0.0.1:8000
uv run --project server pytest server/tests          # tests
```

Endpoints (writes need the header `x-verifier-client: 1`):

| Endpoint | What it does |
|---|---|
| `GET /api/health`, `GET /api/status` | server health; Ollama, sealed-run and running-job status |
| `GET /api/flow` | ready datasets, their kinds, test sizes per gap, eval sizes, measured speeds |
| `GET /api/flow/library?dataset=` | a fixed sample of DEV messages per kind (the page's graph and suggestions) |
| `POST /api/own?name&format&...` | add your own labelled messages (csv/jsonl); checked, then split once |
| `POST /api/agent/check` | send three training messages to your agent and show what came back |
| `POST /api/jobs/quick` | start a quick test: dataset, gap, models, kinds, optional `one_route` and `agent` |
| `GET /api/jobs/{id}`, `GET /api/jobs/{id}/result` | progress, then the plan, scope, verdict and per-message answers |
| `GET /api/story`, `POST /api/jobs/rehearsal` | the older story page and its DEV rehearsal |

Docs at `/api/docs`. Quick tests need Ollama unless they race your own agent. The sealed run is not
offered as a job: it runs once, by hand.
