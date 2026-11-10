"""Run the data endpoints.

    uv run --project mcp_server python -m verifier_mcp            # stdio (Claude Desktop)
    uv run --project mcp_server python -m verifier_mcp --http     # read-only HTTP, 127.0.0.1:8765
    ... --http --allow-writes                                     # HTTP with add_dataset

Stdio always has the write tool: the client that launched it is the owner's. HTTP has no auth,
so it binds to 127.0.0.1 and is read-only unless --allow-writes is given.
"""

import argparse
from pathlib import Path

from verifier_mcp.server import build_server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Agent Verifier MCP data endpoints")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--allow-writes", action="store_true", help="HTTP only: enable write tools")
    ap.add_argument(
        "--preload", action="store_true", help="download the menu's datasets in the background"
    )
    args = ap.parse_args(argv)
    if args.preload and not args.http:
        # stdout is the stdio protocol channel; downloads print progress there
        ap.error("--preload needs --http")
    if args.preload:
        # in the background, so an offline start never blocks or fails; it writes no scenario
        import logging
        import threading

        from agent_committee.catalog import preload

        def run() -> None:
            logging.getLogger("verifier_mcp").warning("preload: %s", preload(args.repo))

        threading.Thread(target=run, daemon=True).start()
    if args.http:
        server = build_server(args.repo, allow_writes=args.allow_writes)
        server.run(transport="http", host=args.host, port=args.port)
    else:
        build_server(args.repo, allow_writes=True).run(transport="stdio")


if __name__ == "__main__":
    main()
