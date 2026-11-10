# Agent Verifier MCP data endpoints. Build from the repository root:
#   docker build -f mcp_server/Dockerfile -t agent-verifier-mcp .
# Run with mcp_server/compose.yaml (the repo is mounted, so splits land in your checkout).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /bin/uv

# The virtualenv lives outside the mounted repo, so it never collides with a host .venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache \
    UV_NO_SYNC=1

WORKDIR /repo
# Only what installing needs; data, sealed/, cache/ and sources/ never enter the image.
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY mcp_server/pyproject.toml mcp_server/uv.lock ./mcp_server/
COPY mcp_server/src ./mcp_server/src
RUN uv sync --project mcp_server --frozen --no-dev

EXPOSE 8765
# Read-only by default; add --allow-writes in compose.yaml to enable add_dataset.
ENTRYPOINT ["uv", "run", "--project", "mcp_server", "--frozen", "--no-sync", "python", "-m", "verifier_mcp", "--repo", "/repo"]
CMD ["--http", "--host", "0.0.0.0", "--port", "8765"]
