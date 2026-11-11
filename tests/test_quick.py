import dataclasses
import hashlib
import json
import zlib
from pathlib import Path

import pytest

from agent_committee import power, quick
from agent_committee.schemas import ArmName, ArmPrediction, Case


def make_cases(n, prefix="c"):
    return [Case(id=f"{prefix}{i:04d}", text=f"text {i}", label=f"l{i % 3}") for i in range(n)]


class Constant:
    def __init__(self, label):
        self.label = label

    def predict(self, case_id, text):
        return ArmPrediction(case_id=case_id, arm=ArmName.LINEAR, pred=self.label)


def test_plan_size_is_the_exact_mcnemar_n_at_the_planning_rate():
    assert quick.PLANNING_DISAGREEMENT == 0.20
    n = quick.plan_size(0.10, 0.05, 0.8)
    assert n == power.required_n(0.10, 0.20, 0.05, 0.8)
    assert quick.plan_size(0.05, 0.05, 0.8) > 3 * n  # halving the gap roughly quadruples N


def test_sample_is_fixed_by_seed_sorted_and_independent_of_input_order():
    cases = make_cases(500)
    a = quick.sample_cases(cases, 100)
    b = quick.sample_cases(list(reversed(cases)), 100)
    assert [c.id for c in a] == [c.id for c in b]
    assert len(a) == 100 and len({c.id for c in a}) == 100
    assert [c.id for c in a] == sorted(c.id for c in a)
    assert quick.sample_cases(cases, 100, seed=1) != a


def test_sample_takes_every_case_when_the_plan_needs_more_than_exist():
    cases = make_cases(40)
    assert quick.sample_cases(cases, 1000) == sorted(cases, key=lambda c: c.id)


def test_comparator_is_the_best_free_model_on_tune_ties_to_the_earlier_rung():
    tune = make_cases(30)  # labels l0, l1, l2 in equal thirds
    best, scores = quick.pick_comparator(
        {"naive_bayes": Constant("l1"), "linear": Constant("l2")}, tune
    )
    assert scores == {"naive_bayes": pytest.approx(1 / 3), "linear": pytest.approx(1 / 3)}
    assert best == "linear"  # tie: linear comes before naive_bayes in CLASSIC
    best, _ = quick.pick_comparator(
        {"majority": Constant("zz"), "naive_bayes": Constant("l0")}, tune
    )
    assert best == "naive_bayes"


def test_plan_file_hash_is_the_hash_of_its_bytes(tmp_path):
    path = tmp_path / "q" / "plan.json"
    sha = quick.write_plan(path, {"gap": 0.05, "cases": 10})
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text(encoding="utf-8")) == {"cases": 10, "gap": 0.05}


def test_estimated_minutes_count_only_llm_arms():
    assert quick.estimate_minutes(600, ["linear", "naive_bayes"]) == 0
    assert quick.estimate_minutes(600, ["prompt", "agent"]) == pytest.approx(600 * 1.34 / 60)


def test_quick_never_imports_the_seal():
    import ast

    tree = ast.parse(Path(quick.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {node.module or ""} | {a.name for a in node.names}
    assert not any(n == "seal" or n.endswith(".seal") for n in names)


@pytest.mark.parametrize(
    "args",
    [
        ["--arms", "linear,naive_bayes"],  # no agent
        ["--arms", "agent,prompt"],  # no free model
        ["--arms", "linear,agent,gpt"],  # unknown arm
    ],
)
def test_main_refuses_arm_sets_without_agent_or_free_model(args, capsys):
    assert quick.main(["--scenario", "banking77", "--gap", "0.1", "--yes", *args]) == 1
    assert "arms must be among" in capsys.readouterr().err


def test_main_refuses_a_gap_that_is_not_a_fraction(capsys):
    assert quick.main(["--scenario", "banking77", "--gap", "5", "--yes"]) == 1
    assert "fraction" in capsys.readouterr().err


class FakeLLMArm:
    def __init__(self, name, accuracy, labels):
        self.name, self.accuracy, self.labels = ArmName(name), accuracy, labels

    def predict(self, case_id, text):
        right = (zlib.crc32(case_id.encode()) % 1000) / 1000 < self.accuracy
        return ArmPrediction(
            case_id=case_id,
            arm=self.name,
            pred=self.labels[case_id] if right else "wrong",
            prompt_tokens=100,
            completion_tokens=5,
            latency_s=0.1,
            own_tool_calls=1,
        )


class FakeClient:
    def __init__(self, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None


def _fake_run(tmp_path, monkeypatch):
    from agent_committee import llm, scenarios
    from agent_committee.arms import agent as agent_arm
    from agent_committee.data import load_dev

    labels = {c.id: c.label for c in load_dev(Path("data"))}
    real = scenarios.get_scenario

    def scenario(name, repo):
        return dataclasses.replace(real(name, repo), results_dir=str(tmp_path))

    monkeypatch.setattr(quick, "get_scenario", scenario)
    monkeypatch.setattr(llm, "LLMClient", FakeClient)
    monkeypatch.setattr(agent_arm, "make_arm", lambda *a, **k: FakeLLMArm("agent", 0.6, labels))


def test_no_cascade_without_linear(tmp_path, monkeypatch):
    """The cascade is derived from linear and the agent; a set without linear must still run."""
    _fake_run(tmp_path, monkeypatch)
    code = quick.main(
        ["--scenario", "banking77", "--gap", "0.2", "--arms", "naive_bayes,agent", "--yes"]
    )
    assert code == 0
    [run] = list((tmp_path / "quick").iterdir())
    verdict = json.loads((run / "verdict.json").read_text(encoding="utf-8"))
    assert "cascade" not in verdict["accuracy"]
    assert (
        json.loads((run / "plan.json").read_text(encoding="utf-8"))["comparator"] == "naive_bayes"
    )


def test_end_to_end_on_banking77_dev_with_fake_llm_arms(tmp_path, monkeypatch):
    """Real DEV data, real free arms; LLM arms faked. Plan first, exploratory verdict, no TEST."""
    _fake_run(tmp_path, monkeypatch)
    code = quick.main(
        ["--scenario", "banking77", "--gap", "0.15", "--arms", "linear,agent", "--yes"]
    )
    assert code == 0
    [run] = list((tmp_path / "quick").iterdir())
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    verdict = json.loads((run / "verdict.json").read_text(encoding="utf-8"))
    assert plan["comparator"] == "linear" and plan["sample_seed"] == quick.SAMPLE_SEED
    assert plan["cases"] == quick.plan_size(0.15, 0.05, 0.8)
    assert verdict["rehearsal"] is True
    assert verdict["reasons"][0].startswith("QUICK TEST")
    plan_sha = hashlib.sha256((run / "plan.json").read_bytes()).hexdigest()
    assert verdict["audit"]["dev_report_sha256"] == f"quick test plan {plan_sha}"
    # a 60% agent against ~90% linear: the "gain ruled out" boundary is crossed early
    assert verdict["verdict"] == "USE_SIMPLER"
    looks = plan["early_stopping"]["looks"]
    assert verdict["n_cases"] in looks and verdict["n_cases"] < plan["cases"]
    assert any(
        r.startswith("Early stopping: a gain of the gap is ruled out") for r in verdict["reasons"]
    )
    cap = next(c for c in verdict["checks"] if c["name"] == "power_cap")
    assert cap["passed"] and cap["low"] == verdict["n_cases"]  # the sequential design governs
    lo, hi = verdict["accuracy_diff_ci"]
    assert hi < 0.15 and verdict["break_even"] == []
    assert (run / "report.html").is_file() and (run / "predictions.jsonl").is_file()


def test_look_order_is_fixed_and_shuffled():
    cases = make_cases(300)
    order = quick.look_order(cases)
    assert [c.id for c in order] == [c.id for c in quick.look_order(list(reversed(cases)))]
    assert sorted(c.id for c in order) == [c.id for c in cases]
    first_look = {c.label for c in order[:60]}
    assert first_look == {"l0", "l1", "l2"} and order != sorted(order, key=lambda c: c.id)


def _verdict(kind):
    from agent_committee.schemas import AuditBundle, EvidenceLevel, Verdict, VerdictJSON

    audit = AuditBundle(
        git_commit="x", prereg_sha256="x", dev_report_sha256="x", test_sha256="x",
        model_digest="x", layer_fingerprint="x", predictions_sha256="x",
    )  # fmt: skip
    return VerdictJSON(
        verdict=Verdict(kind), evidence_level=EvidenceLevel.REAL, audit=audit, reasons=["r"],
        scorer="exact_label", quality_floor_passed=True, accuracy={"agent": 0.8},
        accuracy_diff_ci=(-0.1, -0.01),
    )  # fmt: skip


def _look(outcome, n=100):
    from agent_committee.sequential import Look

    return Look(n=n, diff=-0.08, se0=0.02, se1=0.03, z0=-4, z1=-7, bound=2.5, outcome=outcome)


def test_engine_winner_without_the_boundary_is_not_a_winner():
    out = quick.apply_early_stop(_verdict("USE_SIMPLER"), _look(None, 500), 500)
    assert out.verdict.value == "NEED_MORE_DATA"
    assert "did not agree" in out.reasons[1]


def test_boundary_and_engine_agreeing_keep_the_winner_with_the_repeated_interval():
    out = quick.apply_early_stop(_verdict("USE_SIMPLER"), _look("simpler"), 500)
    assert out.verdict.value == "USE_SIMPLER"
    assert out.accuracy_diff_ci == pytest.approx((-0.08 - 2.5 * 0.02, -0.08 + 2.5 * 0.03))
    assert out.reasons[0].startswith("Early stopping: a gain of the gap is ruled out at 100")


def test_boundary_without_the_engine_is_not_a_winner():
    out = quick.apply_early_stop(_verdict("NEED_MORE_DATA"), _look("simpler"), 500)
    assert out.verdict.value == "NEED_MORE_DATA"


def test_invalid_stays_invalid():
    from agent_committee.schemas import AuditBundle, EvidenceLevel, Verdict, VerdictJSON

    v = VerdictJSON(
        verdict=Verdict.INVALID, evidence_level=EvidenceLevel.REAL,
        audit=AuditBundle(git_commit="x", prereg_sha256="x", dev_report_sha256="x",
                          test_sha256="x", model_digest="x", layer_fingerprint="x"),
        reasons=["floor"], accuracy={},
    )  # fmt: skip
    assert quick.apply_early_stop(v, _look("agent"), 500) is v
