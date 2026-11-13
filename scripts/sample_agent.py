"""A deliberately simple "bring your agent" example (standard library only).

It speaks the page's agent format: POST {"text": "...", "labels": [...]} -> {"label": "..."}.
It picks the label whose words appear most in the message (no training, never sees test answers),
so it is a stand-in for trying the connection, not a good agent.

    uv run python scripts/sample_agent.py            # http://127.0.0.1:8091/classify
    uv run python scripts/sample_agent.py --port 9000
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORD = re.compile(r"[a-z]+")


def stem(w: str) -> str:
    return w[:5]


def pick(text: str, labels: list[str]) -> str:
    words = {stem(w) for w in WORD.findall(text.lower()) if len(w) > 2}
    best, top = "not_mine", 0.0
    for label in labels:
        parts = {stem(w) for w in WORD.findall(label.lower().replace("_", " ")) if len(w) > 2}
        if not parts:
            continue
        score = len(words & parts) / len(parts)
        if score > top:
            best, top = label, score
    return best


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server's name)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            out = {"label": pick(str(body["text"]), list(body.get("labels") or []))}
            code = 200
        except (ValueError, KeyError, TypeError) as exc:
            out, code = {"error": str(exc)}, 400
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="A sample agent for 'Bring your agent'.")
    ap.add_argument("--port", type=int, default=8091)
    port = ap.parse_args().port
    print(f"sample agent on http://127.0.0.1:{port}/classify", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
