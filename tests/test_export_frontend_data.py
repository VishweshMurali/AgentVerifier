import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_committee.llm import LLMClient, LLMUnavailable
from agent_committee.prereg import load_prereg
from agent_committee.schemas import AuditBundle, EvidenceLevel, Verdict, VerdictJSON

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "export_frontend_data.py"
spec = importlib.util.spec_from_file_location("export_frontend_data", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

PREREG_SHA = hashlib.sha256((REPO / "prereg.yaml").read_bytes()).hexdigest()


def verdict(prereg_sha256="p", **kw):
    return VerdictJSON(
        verdict=Verdict.INVALID,
        evidence_level=EvidenceLevel.REAL,
        audit=AuditBundle(
            git_commit="c",
            prereg_sha256=prereg_sha256,
            dev_report_sha256="d",
            test_sha256="t",
            model_digest="m",
            layer_fingerprint="f",
        ),
        reasons=["r"],
        accuracy={},
        **kw,
    )


def test_sealed_verdict_wins_over_the_rehearsal(tmp_path):
    sealed, rehearsal = tmp_path / "v.json", tmp_path / "r.json"
    rehearsal.write_text(verdict(rehearsal=True).model_dump_json(), encoding="utf-8")
    assert export.pick_verdict(sealed, rehearsal)[1] == "rehearsal"
    sealed.write_text(verdict().model_dump_json(), encoding="utf-8")
    assert export.pick_verdict(sealed, rehearsal)[1] == "sealed"


def test_a_rehearsal_is_never_exported_as_sealed(tmp_path):
    sealed = tmp_path / "v.json"
    sealed.write_text(verdict(rehearsal=True).model_dump_json(), encoding="utf-8")
    with pytest.raises(export.ExportRefused, match="rehearsal"):
        export.pick_verdict(sealed, tmp_path / "none.json")


def test_no_verdict_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        export.pick_verdict(tmp_path / "a", tmp_path / "b")


def test_prereg_must_be_the_one_the_verdict_was_judged_under(tmp_path):
    path = REPO / "prereg.yaml"
    assert export.load_checked_prereg(verdict(PREREG_SHA), path) == load_prereg(path)
    edited = tmp_path / "prereg.yaml"
    edited.write_bytes(path.read_bytes() + b"\n# edited\n")
    with pytest.raises(export.ExportRefused, match="judged under"):
        export.load_checked_prereg(verdict(PREREG_SHA), edited)


def test_story_carries_the_verdict_prices_and_rules():
    prereg = load_prereg()
    story = export.build_story(verdict(), "sealed", {"id": "x"}, prereg)
    assert story["verdict"]["verdict"] == "INVALID"
    assert [t["name"] for t in story["price_tables"]] == [t.name for t in prereg.price_tables]
    assert story["rules"]["min_gap"] == prereg.min_gap
    assert story["default_price_table"] in [t.name for t in prereg.price_tables]
    assert story["case"] == {"id": "x"}
    assert story["case_source"] == "dev"
    assert export.build_story(verdict(), "sealed", None, prereg)["case_source"] is None
    json.dumps(story)


def test_default_price_table_must_exist_in_the_prereg():
    prereg = load_prereg()
    renamed = [t.model_copy(update={"name": f"x{i}"}) for i, t in enumerate(prereg.price_tables)]
    with pytest.raises(export.ExportRefused, match="default price table"):
        export.build_story(
            verdict(), "sealed", None, prereg.model_copy(update={"price_tables": renamed})
        )


def test_main_refuses_a_changed_prereg(tmp_path, capsys):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "verdict.json").write_text(verdict("other").model_dump_json())
    (tmp_path / "prereg.yaml").write_bytes((REPO / "prereg.yaml").read_bytes())
    out = tmp_path / "story.json"
    assert export.main(["--root", str(tmp_path), "--out", str(out), "--no-case"]) == 1
    assert not out.exists() and "judged under" in capsys.readouterr().err
    (tmp_path / "results" / "verdict.json").write_text(verdict(PREREG_SHA).model_dump_json())
    assert export.main(["--root", str(tmp_path), "--out", str(out), "--no-case"]) == 0
    assert json.loads(out.read_text())["source"] == "sealed"


# --- the read-only replay -------------------------------------------------------------------

MESSAGES = [{"role": "user", "content": "which label?"}]
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}}]


def fake_transport(request):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": "card_linking"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


def filled_cache(tmp_path):
    db = tmp_path / "llm.sqlite"
    with LLMClient(db, phase="build", model_digest="dg", transport=fake_transport) as client:
        written = client.chat(MESSAGES, arm="agent", tools=TOOLS, max_tokens=50)
    return db, written


def test_cached_replies_return_what_llmclient_stored(tmp_path):
    db, written = filled_cache(tmp_path)
    with export.CachedReplies(db, phase="build", model_digest="dg") as replies:
        got = replies.chat(MESSAGES, arm="agent", tools=TOOLS, max_tokens=50)
    assert got.message == written.message
    assert (got.prompt_tokens, got.completion_tokens, got.finish_reason) == (11, 3, "stop")
    assert got.cached and replies.results == [got]
    with LLMClient(db, phase="build", model_digest="dg", transport=None) as client:
        assert client.chat(MESSAGES, arm="agent", tools=TOOLS, max_tokens=50).message == got.message


@pytest.mark.parametrize(
    "change",
    [
        {"arm": "prompt"},
        {"max_tokens": 51},
        {"tools": None},
    ],
)
def test_cached_replies_miss_on_any_other_request(tmp_path, change):
    db, _ = filled_cache(tmp_path)
    call = {"arm": "agent", "tools": TOOLS, "max_tokens": 50, **change}
    with export.CachedReplies(db, phase="build", model_digest="dg") as replies:
        with pytest.raises(LLMUnavailable):
            replies.chat(MESSAGES, **call)
    for phase, digest in (("run", "dg"), ("build", "other")):
        with export.CachedReplies(db, phase=phase, model_digest=digest) as replies:
            with pytest.raises(LLMUnavailable):
                replies.chat(MESSAGES, arm="agent", tools=TOOLS, max_tokens=50)


def test_cached_replies_never_write(tmp_path):
    db, _ = filled_cache(tmp_path)
    before = db.read_bytes()
    with export.CachedReplies(db, phase="build", model_digest="dg") as replies:
        replies.chat(MESSAGES, arm="agent", tools=TOOLS, max_tokens=50)
        with pytest.raises(LLMUnavailable):
            replies.chat(MESSAGES, arm="other")
        with pytest.raises(Exception, match="readonly"):
            replies._db.execute("DELETE FROM calls")
    assert db.read_bytes() == before


def test_replay_without_a_cache_file_is_no_case_and_creates_nothing(tmp_path):
    assert export.replay_case(tmp_path) is None
    assert not (tmp_path / "cache").exists()


def case(i, label):
    return SimpleNamespace(id=f"c{i}", text=f"text {i}", label=label)


def reply(*queries):
    calls = [{"function": {"arguments": json.dumps({"query": q})}} for q in queries]
    return SimpleNamespace(message={"tool_calls": calls} if calls else {})


def test_first_overruled_picks_the_first_right_answer_the_agent_overturned():
    cases = [case(0, "a"), case(1, "a"), case(2, "a"), case(3, "a"), case(4, "a"), case(5, "a")]
    linear = {"text 0": "b", "text 1": "a", "text 2": "a", "text 3": "a", "text 4": "a"}
    results: list = []

    def predict(case_id, text):
        if case_id == "c1":
            raise LLMUnavailable("cache miss")
        # replies pile up across cases; the query must come from this case's first reply
        results.extend([reply(f"q {case_id}"), reply("later")])
        answers = {"c2": "a", "c3": None, "c4": "z"}
        return SimpleNamespace(pred=answers.get(case_id, "z"))

    def decide(text):
        return SimpleNamespace(choice=linear.get(text, "x"), shortlist=("a", "z"))

    got = export.first_overruled(cases, decide, predict, results)
    assert got == {
        "id": "c4",
        "text": "text 4",
        "gold": "a",
        "linear": "a",
        "shortlist": ["a", "z"],
        "search_query": "q c4",
        "agent": "z",
    }
    assert export.first_overruled(cases[:4], decide, predict, results) is None


@pytest.mark.skipif(
    not (REPO / "cache" / "llm.sqlite").is_file(), reason="needs the local LLM cache"
)
def test_replay_on_the_real_cache_is_deterministic_and_read_only():
    db = REPO / "cache" / "llm.sqlite"
    before = (db.stat().st_mtime_ns, db.stat().st_size)
    first = export.replay_case(REPO)
    assert first is not None and first["linear"] == first["gold"] != first["agent"]
    assert (db.stat().st_mtime_ns, db.stat().st_size) == before
