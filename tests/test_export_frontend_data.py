import importlib.util
import json
from pathlib import Path

import pytest

from agent_committee.prereg import load_prereg
from agent_committee.schemas import AuditBundle, EvidenceLevel, Verdict, VerdictJSON

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_frontend_data.py"
spec = importlib.util.spec_from_file_location("export_frontend_data", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def verdict(**kw):
    return VerdictJSON(
        verdict=Verdict.INVALID,
        evidence_level=EvidenceLevel.REAL,
        audit=AuditBundle(
            git_commit="c",
            prereg_sha256="p",
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


def test_no_verdict_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        export.pick_verdict(tmp_path / "a", tmp_path / "b")


def test_story_carries_the_verdict_prices_and_rules():
    prereg = load_prereg()
    story = export.build_story(verdict(), "sealed", {"id": "x"}, prereg)
    assert story["verdict"]["verdict"] == "INVALID"
    assert [t["name"] for t in story["price_tables"]] == [t.name for t in prereg.price_tables]
    assert story["rules"]["min_gap"] == prereg.min_gap
    assert story["case"] == {"id": "x"}
    json.dumps(story)
