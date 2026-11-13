"""The guide's conversation (MOCKUP, owner review; bends the locked "no AI text on the page"
rule, so it stays behind /api/guide and the mockup until the owner approves it).

Plain code works out every fact (the page sends the facts it computed from the run); the local
model only words a reply about the reader's own agent from those facts. Before the reply is
shown, plain code checks it: every number in it must appear among the facts, or the reply is
replaced by the plain-code answer. The model never scores, decides or adds a number.

Calls go straight to Ollama (not through agent_committee.llm), so guide traffic never enters the
arms' cache that the leakage check scans.
"""

import json
import re
import urllib.request
from typing import Any

from agent_committee.llm import BASE_URL, MODEL

TIMEOUT_S = 60.0
MAX_FACTS = 6000
SYSTEM = (
    "You are a friendly, sharp guide on a website that tests whether a business needs an AI "
    "agent. Talk to the reader about THEIR agent (described below), in 3 to 5 short sentences. "
    "Explain the WHY: name the specific mix-ups and quote one real message from the facts when it "
    "helps, then end with one concrete next step they could look at. Use ONLY the facts given. "
    "Never invent, round differently or compute a number; copy numbers exactly as written in the "
    "facts, or leave them out. If the facts do not answer the question, say so in one sentence. "
    "No lists, no markdown."
)

_NUM = re.compile(r"(?<![\w.])[-−]?\$?\d[\d,]*(?:\.\d+)?%?")


def numbers(text: str) -> set[str]:
    """Numbers as written, normalised: '$4,747' -> '4747', '97%' -> '97', '−2.1' -> '-2.1'."""
    out = set()
    for m in _NUM.findall(text):
        n = m.replace("−", "-").replace("$", "").replace(",", "").rstrip("%")
        if n not in ("", "-"):
            out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


_PAIR = re.compile(r"(\d[\d,]*)\s+(?:out\s+)?of\s+(?:the\s+)?(\d[\d,]*)", re.IGNORECASE)


def pairs(text: str) -> set[tuple[str, str]]:
    """'3 of 95', '1 out of 20' -> {('3', '95'), ('1', '20')}: counts must match as a pair."""
    return {(a.replace(",", ""), b.replace(",", "")) for a, b in _PAIR.findall(text)}


def grounded(reply: str, facts: str, question: str) -> tuple[bool, list[str]]:
    """True when every number in the reply appears in the facts or the reader's own question.
    Small counting words the model may add (1-5, e.g. 'two sentences') are allowed."""
    allowed = numbers(facts) | numbers(question) | {str(i) for i in range(6)}
    allowed |= {a.lstrip("-") for a in allowed}
    bad = sorted(n for n in numbers(reply) if n not in allowed and n.lstrip("-") not in allowed)
    # a count is only true with its total: "1 of 20" must be a pair the run states
    known = pairs(facts) | pairs(question)
    bad += sorted(f"{a} of {b}" for a, b in pairs(reply) if (a, b) not in known)
    return (not bad, bad)


def ask_model(question: str, description: str, facts: str, call=None) -> str:
    """One chat call at temperature 0 with a fixed seed (same question, same facts, same reply)."""
    body = {
        "model": MODEL,
        "temperature": 0,
        "seed": 0,
        "max_tokens": 220,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"The reader's agent: {description or '(not described)'}\n\n"
                f"Facts from their test run:\n{facts[:MAX_FACTS]}\n\nQuestion: {question}",
            },
        ],
    }
    if call is not None:
        return call(body)
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310 (local Ollama)
        data = json.loads(r.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"]).strip()


def guide_reply(payload: dict[str, Any], call=None) -> dict[str, Any]:
    """{question, description, facts, fallback} -> {reply, worded_by, checked, rejected}."""
    question = str(payload.get("question", ""))[:500]
    description = str(payload.get("description", ""))[:300]
    facts = str(payload.get("facts", ""))
    fallback = str(payload.get("fallback", ""))
    if not question.strip() or not facts.strip():
        return {"reply": fallback, "worded_by": "code", "checked": True, "rejected": []}
    try:
        reply = ask_model(question, description, facts, call)
    except Exception as exc:  # Ollama down or slow: the plain answer still stands
        return {
            "reply": fallback,
            "worded_by": "code",
            "checked": True,
            "rejected": [],
            "error": str(exc)[:200],
        }
    ok, bad = grounded(reply, facts, question)
    if not ok or not reply:
        return {"reply": fallback, "worded_by": "code", "checked": True, "rejected": bad}
    return {"reply": reply, "worded_by": "model", "checked": True, "rejected": []}
