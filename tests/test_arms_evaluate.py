import json

import pytest

from agent_committee.arms import evaluate
from agent_committee.schemas import ArmName, ArmPrediction, Case


class _Fixed:
    name = ArmName.PROMPT

    def __init__(self, answers):
        self.answers = answers

    def fit(self, pool):  # pragma: no cover
        pass

    def predict(self, case_id, text):
        return ArmPrediction(case_id=case_id, arm=self.name, pred=self.answers[case_id])


def test_score_counts_unparseable_as_wrong():
    cases = [Case(id=str(i), text="t", label="x") for i in range(4)]
    r = evaluate.score(_Fixed({"0": "x", "1": "x", "2": "y", "3": None}), cases)
    assert r["correct"] == 2 and r["accuracy"] == 0.5 and r["unparseable"] == 1


def test_best_simpler_prefers_higher_then_tfidf():
    assert (
        evaluate.best_simpler({"tfidf": {"accuracy": 0.9}, "prompt": {"accuracy": 0.8}}) == "tfidf"
    )
    assert (
        evaluate.best_simpler({"tfidf": {"accuracy": 0.7}, "prompt": {"accuracy": 0.8}}) == "prompt"
    )
    assert (
        evaluate.best_simpler({"tfidf": {"accuracy": 0.8}, "prompt": {"accuracy": 0.8}}) == "tfidf"
    )
    assert evaluate.best_simpler({"agent": {"accuracy": 1.0}}) is None


def test_write_report_refuses_overwrite(tmp_path):
    path = tmp_path / "r.json"
    report = evaluate.build_report({"tfidf": {"accuracy": 0.5}}, {"pool": 1})
    evaluate.write_report(path, report)
    assert json.loads(path.read_text())["best_simpler_arm"] == "tfidf"
    with pytest.raises(FileExistsError):
        evaluate.write_report(path, report)


def test_cli_refuses_existing_out(tmp_path):
    out = tmp_path / "r.json"
    out.write_text("{}")
    assert evaluate.main(["--out", str(out)]) == 1
