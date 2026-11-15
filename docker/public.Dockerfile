# The public live demo as ONE standalone image (page + API): nothing is mounted at run time.
# Build from the repository root:   docker build -f docker/public.Dockerfile -t agent-verifier .
# Run it the way a host would:
#   docker run --rm -p 8000:8000 -e PORT=8000 -e GROQ_API_KEY=... agent-verifier
# render.yaml deploys this file. The local setup (compose.yaml, Ollama on the host) keeps using
# docker/web.Dockerfile.
#
# What is inside: the code, prereg.yaml, the scenario files, the DEV splits and the test HASHES.
# What never is: the sealed test sets, any LLM cache, results, uploads (.dockerignore).

# --- 1. build the page (server mode) -------------------------------------------------------
FROM node:24-slim AS page
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build:server

# --- 2. the API server ----------------------------------------------------------------------
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache \
    UV_NO_SYNC=1 \
    VERIFIER_PUBLIC=1 \
    AGENT_PROVIDER=groq \
    VERIFIER_HOST=0.0.0.0 \
    FALSE_ALARM_WORKERS=1

WORKDIR /repo
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY server/pyproject.toml server/uv.lock ./server/
COPY server/src ./server/src
RUN uv sync --project server --frozen --no-dev

# what the server reads at run time (the story export script, costs, scenarios, DEV data)
COPY scripts/export_frontend_data.py ./scripts/export_frontend_data.py
COPY prereg.yaml ./
COPY scenarios ./scenarios
COPY data ./data

# train the light free models once, here, so a visitor's first test only waits for the agent
# (the heavy tree models are not offered on the public site)
RUN uv run --project server --frozen --no-sync python -m agent_committee.warm \
    --all --arms majority,naive_bayes,linear_svm --repo /repo

COPY --from=page /build/frontend/dist-server /app/page

EXPOSE 8000
# the host names the port in PORT (Render: 10000); GROQ_API_KEY comes from the host's secrets
ENTRYPOINT ["uv", "run", "--project", "server", "--frozen", "--no-sync", "python", "-m", "verifier_server"]
CMD ["--root", "/repo", "--page-dir", "/app/page", "--no-case"]
