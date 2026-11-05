"""Run the local API and page:  uv run --project server python -m verifier_server [--port 8000]"""

import argparse

import uvicorn

from .app import create_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    # local only: never bind to other interfaces
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
