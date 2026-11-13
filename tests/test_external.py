import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_committee.external import (
    AgentDown,
    ExternalAgentArm,
    check_url,
    public_endpoint,
)


def serve(answer):
    """A tiny agent on a free local port; `answer(text, labels)` gives the JSON reply."""

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            out = json.dumps(answer(body["text"], body["labels"])).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/classify"


def test_only_http_addresses_with_a_host():
    assert check_url(" http://x:9/a ") == "http://x:9/a"
    for bad in ("ftp://x/a", "file:///etc/passwd", "localhost:9", "http:///a"):
        with pytest.raises(ValueError):
            check_url(bad)


def test_saved_address_drops_keys_and_credentials():
    assert (
        public_endpoint("https://u:p@api.x.com:8443/v1/c?key=SECRET")
        == "https://api.x.com:8443/v1/c"
    )


def test_valid_labels_count_and_junk_is_no_answer():
    srv, url = serve(lambda text, labels: {"label": text})
    try:
        arm = ExternalAgentArm(url, ["card_arrival", "lost_card"])
        ok = arm.predict("1", "card_arrival")
        assert ok.pred == "card_arrival" and ok.own_tool_calls == 1 and not ok.malformed
        assert arm.predict("2", "not_mine").pred == "not_mine"
        junk = arm.predict("3", "pizza")  # never mapped to not_mine and counted right
        assert junk.pred is None and junk.malformed and junk.own_tool_calls == 0
    finally:
        srv.shutdown()


def test_the_agent_gets_the_labels_and_a_non_object_reply_is_an_error():
    seen = {}

    def answer(text, labels):
        seen["labels"] = labels
        return ["not", "an", "object"]

    srv, url = serve(answer)
    try:
        with pytest.raises(AgentDown):
            ExternalAgentArm(url, ["b", "a"], retries=0).predict("1", "hi")
        assert seen["labels"] == ["a", "b"]
    finally:
        srv.shutdown()


def test_a_dead_agent_stops_the_run():
    srv, url = serve(lambda text, labels: {"label": "a"})
    srv.shutdown()
    srv.server_close()
    with pytest.raises(AgentDown, match="stopped answering"):
        ExternalAgentArm(url, ["a"], retries=0).predict("1", "hi")
