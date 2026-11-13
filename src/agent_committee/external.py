"""The reader's own agent, plugged in over HTTP for quick tests (owner request 2026-11-12).

Contract: POST <url> with JSON {"text": "<one message>", "labels": [<the dataset's kinds>]};
the agent answers JSON {"label": "<one of the labels>"} (or "not_mine"). It takes the agent slot
of a quick test, so it is compared, checked and decided exactly like the reference agent.

Honesty rules:
- only the dataset's own labels (or not_mine) count as answers; anything else is no answer and
  scores wrong, so junk can never be mapped to "not mine" and counted right;
- an agent that stops answering stops the run (like Ollama being down): a dead endpoint must not
  turn into a confident verdict;
- the agent is not pinned, so a re-run may differ; plan.json and the audit say so;
- its cost is the price per message the reader states (no tokens are seen).

Standard library only (urllib). Not part of the frozen arms; the sealed runner never uses it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable

from agent_committee.schemas import ArmName, ArmPrediction

NOT_MINE = "not_mine"
TIMEOUT_S = 30.0
RETRIES = 2


class AgentDown(RuntimeError):
    """The reader's agent stopped answering; the run stops rather than scoring it wrong."""


def check_url(url: str) -> str:
    """The URL if it is plain http(s) with a host, else ValueError."""
    p = urllib.parse.urlsplit(url.strip())
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("the agent's address must start with http:// or https:// and name a host")
    return url.strip()


def public_endpoint(url: str) -> str:
    """The address without its query string or credentials (they may hold a key): safe to save."""
    p = urllib.parse.urlsplit(url)
    host = p.hostname or ""
    if p.port:
        host = f"{host}:{p.port}"
    return urllib.parse.urlunsplit((p.scheme, host, p.path, "", ""))


def ask(url: str, text: str, labels: list[str], timeout: float = TIMEOUT_S) -> tuple[object, float]:
    """One call: (the agent's "label" field, seconds). Raises URLError/ValueError on failure."""
    body = json.dumps({"text": text, "labels": labels}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST", headers={"content-type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (http/https checked)
        raw = r.read(65536)
    took = time.perf_counter() - t0
    out = json.loads(raw.decode("utf-8"))
    if not isinstance(out, dict):
        raise ValueError('the agent must answer a JSON object like {"label": "..."}')
    return out.get("label"), took


class ExternalAgentArm:
    """The reader's agent in the agent slot. `own_tool_calls` is 1 for a valid answer, so the
    quality floor reads as the share of valid answers (the only thing a black box shows)."""

    name = ArmName.AGENT

    def __init__(self, url: str, labels: Iterable[str], retries: int = RETRIES) -> None:
        self.url = check_url(url)
        self.labels = sorted(set(labels))
        self.valid = set(self.labels) | {NOT_MINE}
        self.retries = retries

    def fit(self, pool) -> None:  # the reader's agent is already built
        return None

    def predict(self, case_id: str, text: str) -> ArmPrediction:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                label, took = ask(self.url, text, self.labels)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
                last = exc
                time.sleep(0.5 * (attempt + 1))
        else:
            raise AgentDown(
                f"your agent at {public_endpoint(self.url)} stopped answering ({last}); "
                "the test stopped rather than score it wrong"
            )
        ok = isinstance(label, str) and label in self.valid
        return ArmPrediction(
            case_id=case_id,
            arm=self.name,
            pred=label if ok else None,
            latency_s=took,
            own_tool_calls=1 if ok else 0,
            malformed=not ok,
        )
