import json

import pytest

from agent_committee.arms import evaluate
from agent_committee.arms.linear import Decision, make_layer
from agent_committee.schemas import ArmName, ArmPrediction, Case


def _decision(null=False, shortlist=("x", "y")):
    return Decision(choice=shortlist[0], score=0.5, null=null, shortlist=shortlist)


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
    arm = _Fixed({"0": "x", "1": "x", "2": "y", "3": None})
    r = evaluate.score(arm, cases, [_decision()] * 4)
    assert r["correct"] == 2 and r["accuracy"] == 0.5 and r["unparseable"] == 1


def test_a_flag_changes_no_score_and_flagged_accuracy_is_reported():
    cases = [Case(id=str(i), text="t", label="x") for i in range(4)]
    arm = _Fixed({"0": "x", "1": "y", "2": "x", "3": "x"})
    flags = [_decision(null=True), _decision(null=True), _decision(), _decision()]
    r = evaluate.score(arm, cases, flags)
    assert r["flagged_cases"] == 2 and r["flagged_correct"] == 1
    assert r["flagged_accuracy"] == 0.5
    assert r["accuracy"] == evaluate.score(arm, cases, [_decision()] * 4)["accuracy"] == 0.75


def test_score_with_no_flagged_cases_has_no_flagged_accuracy():
    cases = [Case(id="0", text="t", label="x")]
    r = evaluate.score(_Fixed({"0": "x"}), cases, [_decision()])
    assert r["flagged_cases"] == 0 and r["flagged_accuracy"] is None


def test_score_reports_own_tool_and_send_back_rates():
    class _Agent(_Fixed):
        def predict(self, case_id, text):
            return ArmPrediction(
                case_id=case_id,
                arm=self.name,
                pred="x",
                own_tool_calls=2 if case_id in ("0", "1") else 0,
                sent_back=case_id == "1",
                malformed=case_id == "2",
            )

    cases = [Case(id=str(i), text="t", label="x") for i in range(4)]
    r = evaluate.score(_Agent({}), cases, [_decision()] * 4)
    # case 1 used a tool only after being sent back, so it is not own tool use
    assert r["own_tool_rate"] == 0.25 and r["send_back_rate"] == 0.25
    assert r["malformed_rate"] == 0.25


def test_score_rejects_misaligned_decisions():
    cases = [Case(id="0", text="t", label="x")]
    with pytest.raises(ValueError):
        evaluate.score(_Fixed({"0": "x"}), cases, [])


def test_layer_report_counts_flags_and_coverage():
    pool = [
        Case(id=f"{x}{i}", text=f"{x} word {i}", label=x) for x in ("aa", "bb") for i in range(4)
    ]
    layer = make_layer(pool)
    cases = [Case(id=str(i), text="t", label=lab) for i, lab in enumerate(("x", "y", "x", "z"))]
    decisions = [_decision(null=True), _decision(), _decision(), _decision()]
    r = evaluate.layer_report(layer, cases, decisions)
    assert r["flagged_rate"] == 0.25 and r["shortlist_coverage"] == 0.75
    assert r["fingerprint"] == layer.fingerprint()
    assert r["shortlist_k"] == layer.shortlist_k and r["null_threshold"] == layer.threshold


def test_best_simpler_prefers_higher_then_linear():
    pick = evaluate.best_simpler
    assert pick({"linear": {"accuracy": 0.9}, "prompt": {"accuracy": 0.8}}) == "linear"
    assert pick({"linear": {"accuracy": 0.7}, "prompt": {"accuracy": 0.8}}) == "prompt"
    assert pick({"linear": {"accuracy": 0.8}, "prompt": {"accuracy": 0.8}}) == "linear"
    assert evaluate.best_simpler({"agent": {"accuracy": 1.0}}) is None


def test_write_report_refuses_overwrite(tmp_path):
    path = tmp_path / "r.json"
    report = evaluate.build_report({"linear": {"accuracy": 0.5}}, {"pool": 1}, {"fingerprint": "f"})
    evaluate.write_report(path, report)
    written = json.loads(path.read_text())
    assert written["best_simpler_arm"] == "linear" and written["layer"]["fingerprint"] == "f"
    with pytest.raises(FileExistsError):
        evaluate.write_report(path, report)


def test_cli_refuses_existing_out(tmp_path):
    out = tmp_path / "r.json"
    out.write_text("{}")
    assert evaluate.main(["--out", str(out)]) == 1
