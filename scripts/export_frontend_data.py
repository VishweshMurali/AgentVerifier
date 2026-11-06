"""Export the pipeline's results for the front end: frontend/public/data/story.json.

    uv run python scripts/export_frontend_data.py [--root .] [--out frontend/public/data/story.json]

Uses the sealed verdict (results/verdict.json) when it exists, otherwise the DEV rehearsal
(results/rehearsal/verdict.json), and says which. Adds the prereg's price tables and rules, and one
DEV case where the agent overruled a right answer, replayed from the build-phase LLM cache. The
replay opens the cache read-only: no model calls and no writes, so it never locks a running job.
Every number the page shows comes from this file; nothing is computed by an LLM.

The export refuses (ExportRefused) a sealed-path verdict marked as a rehearsal, a prereg.yaml whose
hash differs from the one the verdict was judged under, and a default price table the prereg lacks.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from agent_committee.prereg import load_prereg
from agent_committee.schemas import Prereg, VerdictJSON

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT / "results" / "verdict.json"
REHEARSAL = ROOT / "results" / "rehearsal" / "verdict.json"
OUT = ROOT / "frontend" / "public" / "data" / "story.json"
CACHE = Path("cache") / "llm.sqlite"
DEFAULT_PRICE_TABLE = "cloud_small"
CASE_SOURCE = "dev"  # the replayed case always comes from the DEV eval split, never the test


class ExportRefused(ValueError):
    """The inputs disagree with each other; exporting them would show the page wrong numbers."""


def pick_verdict(sealed: Path = SEALED, rehearsal: Path = REHEARSAL) -> tuple[VerdictJSON, str]:
    for path, source in ((sealed, "sealed"), (rehearsal, "rehearsal")):
        if path.exists():
            verdict = VerdictJSON.model_validate_json(path.read_text(encoding="utf-8"))
            if source == "sealed" and verdict.rehearsal:
                raise ExportRefused(f"{path} is a rehearsal verdict; it cannot be shown as sealed")
            return verdict, source
    raise FileNotFoundError("no verdict yet: run the rehearsal or the sealed run first")


def load_checked_prereg(verdict: VerdictJSON, path: Path = ROOT / "prereg.yaml") -> Prereg:
    """prereg.yaml, only if it is byte-identical to the one the verdict was judged under."""
    found = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if found != verdict.audit.prereg_sha256:
        raise ExportRefused(
            f"{path} has sha256 {found}, but the verdict was judged under "
            f"{verdict.audit.prereg_sha256}: its prices and rules would not match the verdict"
        )
    return load_prereg(path)


class CachedReplies:
    """`LLMClient.chat` answered from the SQLite cache alone, opened read-only.

    Keys are built exactly as `LLMClient.chat` builds them, so a hit returns the same stored
    answer. A miss raises `LLMUnavailable`; nothing is ever written (no cache rows, no call log).
    Every result is kept in `results` in call order.
    """

    def __init__(self, db_path: Path, *, phase: str, model_digest: str | None) -> None:
        from agent_committee.llm import MODEL

        self.phase, self.model, self.model_digest = phase, MODEL, model_digest
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        self._db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.results: list = []

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "CachedReplies":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        arm: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = 0,
    ):
        from agent_committee.llm import LLMResult, LLMUnavailable, request_key

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            request["tools"] = tools
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if seed is not None:
            request["seed"] = seed
        keyed = {**request, "_phase": self.phase, "_arm": arm, "_digest": self.model_digest}
        row = self._db.execute(
            "SELECT response_json, prompt_tokens, completion_tokens, latency_s, finish_reason "
            "FROM cache WHERE key = ?",
            (request_key(keyed),),
        ).fetchone()
        if row is None:
            raise LLMUnavailable("cache miss: the export makes no model calls")
        result = LLMResult(json.loads(row[0]), row[1], row[2], row[3], True, 0, row[4])
        self.results.append(result)
        return result


def first_overruled(
    cases: Iterable, decide: Callable, predict: Callable, results: list
) -> dict | None:
    """First case where linear was right and the agent answered a different label.

    `results` is the list the replies land in; it is cleared before each case so the search
    query read from it belongs to that case. Cases the cache cannot answer are skipped.
    """
    from agent_committee.llm import LLMUnavailable

    for case in cases:
        decision = decide(case.text)
        if decision.choice != case.label:
            continue
        results.clear()
        try:
            pred = predict(case.id, case.text)
        except LLMUnavailable:
            continue
        if pred.pred == case.label or pred.pred is None:
            continue
        tool_calls = [c for r in results for c in (r.message.get("tool_calls") or [])]
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


def replay_case(root: Path = ROOT) -> dict | None:
    """First DEV eval case where linear was right and the agent wrong, from the cache only.

    Reads `root/data` and `root/cache/llm.sqlite` (read-only). No cache file means no case.
    """
    from agent_committee.arms import agent as agent_arm
    from agent_committee.arms.config import MODEL_DIGEST
    from agent_committee.arms.labels import build_label_book
    from agent_committee.arms.linear import make_layer
    from agent_committee.arms.splits import make_arms_split
    from agent_committee.data import load_dev

    db_path = Path(root) / CACHE
    if not db_path.is_file():
        return None
    split = make_arms_split(load_dev(Path(root) / "data"))
    book = build_label_book(split.pool)
    layer = make_layer(split.pool)
    with CachedReplies(db_path, phase="build", model_digest=MODEL_DIGEST) as replies:
        arm = agent_arm.make_arm(split.pool, book, layer, replies)
        return first_overruled(split.eval, layer.decide, arm.predict, replies.results)


def build_story(verdict: VerdictJSON, source: str, case: dict | None, prereg) -> dict:
    names = [t.name for t in prereg.price_tables]
    if DEFAULT_PRICE_TABLE not in names:
        raise ExportRefused(f"default price table {DEFAULT_PRICE_TABLE!r} is not in {names}")
    return {
        "source": source,
        "verdict": json.loads(verdict.model_dump_json()),
        "price_tables": [t.model_dump() for t in prereg.price_tables],
        "default_price_table": DEFAULT_PRICE_TABLE,
        "rules": {
            "alpha": prereg.alpha,
            "min_gap": prereg.min_gap,
            "checks": len(verdict.checks),
            "price_tables": len(prereg.price_tables),
        },
        "case": case,
        "case_source": CASE_SOURCE if case is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path, default=ROOT, help="folder holding results/, data/, cache/"
    )
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-case", action="store_true", help="skip the cached case replay")
    args = ap.parse_args(argv)
    root = args.root
    try:
        verdict, source = pick_verdict(
            root / "results" / "verdict.json", root / "results" / "rehearsal" / "verdict.json"
        )
        prereg = load_checked_prereg(verdict, root / "prereg.yaml")
        case = None if args.no_case else replay_case(root)
        story = build_story(verdict, source, case, prereg)
    except (FileNotFoundError, ExportRefused) as exc:
        print(exc, file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(story, indent=2) + "\n", encoding="utf-8")
    print(f"{source} data -> {args.out}" + ("" if case else " (no case trace)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
