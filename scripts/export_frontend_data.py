"""Export the pipeline's results for the front end: frontend/public/data/story.json.

    uv run python scripts/export_frontend_data.py [--out frontend/public/data/story.json]

Uses the sealed verdict (results/verdict.json) when it exists, otherwise the DEV rehearsal
(results/rehearsal/verdict.json), and says which. Adds the prereg's price tables and rules, and one
DEV case where the agent overruled a right answer, replayed from the build-phase LLM cache (no new
model calls). Every number the page shows comes from this file; nothing is computed by an LLM.
"""

import argparse
import json
import sys
from pathlib import Path

from agent_committee.prereg import load_prereg
from agent_committee.schemas import VerdictJSON

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT / "results" / "verdict.json"
REHEARSAL = ROOT / "results" / "rehearsal" / "verdict.json"
OUT = ROOT / "frontend" / "public" / "data" / "story.json"


def pick_verdict(sealed: Path = SEALED, rehearsal: Path = REHEARSAL) -> tuple[VerdictJSON, str]:
    for path, source in ((sealed, "sealed"), (rehearsal, "rehearsal")):
        if path.exists():
            return VerdictJSON.model_validate_json(path.read_text(encoding="utf-8")), source
    raise FileNotFoundError("no verdict yet: run the rehearsal or the sealed run first")


def replay_case() -> dict | None:
    """First DEV eval case where linear was right and the agent wrong, from the cache only."""
    from agent_committee.arms import agent as agent_arm
    from agent_committee.arms.config import MODEL_DIGEST
    from agent_committee.arms.labels import build_label_book
    from agent_committee.arms.linear import make_layer
    from agent_committee.arms.splits import make_arms_split
    from agent_committee.data import load_dev
    from agent_committee.llm import LLMClient, LLMUnavailable

    def offline(request):
        raise LLMUnavailable("cache miss: the export makes no model calls")

    split = make_arms_split(load_dev(ROOT / "data"))
    book = build_label_book(split.pool)
    layer = make_layer(split.pool)
    with LLMClient(
        ROOT / "cache" / "llm.sqlite", phase="build", model_digest=MODEL_DIGEST, transport=offline
    ) as client:
        arm = agent_arm.make_arm(split.pool, book, layer, client)
        calls: list = []
        real_chat = client.chat

        def spy(*args, **kwargs):
            result = real_chat(*args, **kwargs)
            calls.append(result)
            return result

        client.chat = spy
        for case in split.eval:
            decision = layer.decide(case.text)
            if decision.choice != case.label:
                continue
            calls.clear()
            try:
                pred = arm.predict(case.id, case.text)
            except LLMUnavailable:
                continue
            if pred.pred == case.label or pred.pred is None:
                continue
            tool_calls = [c for r in calls for c in (r.message.get("tool_calls") or [])]
            query = None
            if tool_calls:
                query = json.loads(tool_calls[0]["function"]["arguments"]).get("query")
            return {
                "id": case.id,
                "text": case.text,
                "gold": case.label,
                "linear": decision.choice,
                "shortlist": list(decision.shortlist),
                "search_query": query,
                "agent": pred.pred,
            }
    return None


def build_story(verdict: VerdictJSON, source: str, case: dict | None, prereg) -> dict:
    return {
        "source": source,
        "verdict": json.loads(verdict.model_dump_json()),
        "price_tables": [t.model_dump() for t in prereg.price_tables],
        "default_price_table": "cloud_small",
        "rules": {
            "alpha": prereg.alpha,
            "min_gap": prereg.min_gap,
            "checks": len(verdict.checks),
            "price_tables": len(prereg.price_tables),
        },
        "case": case,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-case", action="store_true", help="skip the cached case replay")
    args = ap.parse_args(argv)
    try:
        verdict, source = pick_verdict()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    case = None if args.no_case else replay_case()
    story = build_story(verdict, source, case, load_prereg(ROOT / "prereg.yaml"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(story, indent=2) + "\n", encoding="utf-8")
    print(f"{source} data -> {args.out}" + ("" if case else " (no case trace)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
