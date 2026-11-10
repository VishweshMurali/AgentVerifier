# The page + its API (server/) as a container. Build from the repository root:
#   docker compose up --build        (see compose.yaml)
# Ollama stays on the host; OLLAMA_BASE_URL points the container at it.

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
# git: rehearsals record the commit; the repo is mounted from the Windows host, so git must
# trust the mount and read CRLF files the way the host does (else every file looks modified)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /repo \
    && git config --system core.autocrlf true

# UV_NO_SYNC: the page starts rehearsals with `uv run`; a sync would rewrite /opt/venv for the
# root project and remove the server's own packages
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache \
    UV_NO_SYNC=1

WORKDIR /repo
# only what installing needs; the repo itself is mounted at run time
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY server/pyproject.toml server/uv.lock ./server/
COPY server/src ./server/src
RUN uv sync --project server --frozen --no-dev

# the built page lives OUTSIDE /repo, so the repo mount cannot hide it
COPY --from=page /build/frontend/dist-server /app/page

EXPOSE 8000
ENTRYPOINT ["uv", "run", "--project", "server", "--frozen", "--no-sync", "python", "-m", "verifier_server"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--root", "/repo", "--page-dir", "/app/page"]
