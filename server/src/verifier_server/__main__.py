"""Run the local API and page:  uv run --project server python -m verifier_server [--port 8000]

--root, --page-dir and --no-case exist for the end-to-end test (a fixture results folder, no
case replay from the LLM cache).
"""

import argparse
from pathlib import Path

import uvicorn

from .app import ROOT, create_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--root", type=Path, default=ROOT, help="folder holding results/ and data/")
    ap.add_argument("--page-dir", type=Path, default=None, help="built page to serve at /")
    ap.add_argument("--no-case", action="store_true", help="skip the case replay (no LLM cache)")
    args = ap.parse_args()
    app = create_app(args.root, page_dir=args.page_dir, with_case=not args.no_case)
    # local only: never bind to other interfaces
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
