import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_committee import runner
from agent_committee.arms.evaluate import SIMPLER as _SIMPLER
from agent_committee.prereg import load_prereg
from agent_committee.schemas import ArmName, ArmPrediction, Case, Verdict

BASE = load_prereg()
PREREG = BASE.model_copy(
    update={
        "stats": BASE.stats.model_copy(update={"bootstrap_replicates": 300, "aa_simulations": 100})
    }
)
WORDS = "card transfer money account balance fee refund top up pin cash atm limit app".split()


def cases(n, prefix, seed):
    rng = np.random.default_rng(seed)
    return [
        Case(id=f"{prefix}{i}", text=" ".join(rng.choice(WORDS, size=7)), label=f"l{i % 4}")
        for i in range(n)
    ]


class FakeArm:
    """Right on a fixed share of cases, with fixed token use and agent behaviour."""

    def __init__(self, name, accuracy, tokens=0, own_tools=True):
        self.name = ArmName(name)
        self.accuracy = accuracy
        self.tokens = tokens
        self.own_tools = own_tools
        self.labels = {}

    def predict(self, case_id, text):
        right = (zlib.crc32(case_id.encode()) % 1000) / 1000 < self.accuracy
        return ArmPrediction(
            case_id=case_id,
            arm=self.name,
            pred=self.labels[case_id] if right else "wrong",
            prompt_tokens=self.tokens,
            completion_tokens=self.tokens // 20,
            latency_s=0.1,
            own_tool_calls=1 if self.own_tools else 0,
            sent_back=not self.own_tools,
        )


class FakeLayer:
    def decide(self, text):
        score = (sum(map(ord, text)) % 100) / 100
        return SimpleNamespace(score=score, null=score < 0.1, choice="l0", shortlist=["l0", "l1"])


def arms_for(test, **agent_kw):
    arms = {
        "linear": FakeArm("linear", 0.9),
        "prompt": FakeArm("prompt", 0.75, tokens=230),
        "agent": FakeArm("agent", 0.82, tokens=700, **agent_kw),
    }
    for a in arms.values():
        a.labels = {c.id: c.label for c in test}
    return arms


DEV_REPORT = {
    "best_simpler_arm": "linear",
    "results": {"linear": {"correct": 180, "n": 200}, "agent": {"correct": 165, "n": 200}},
}
AUDIT = {
    "git_commit": "c",
    "prereg_sha256": "p",
    "dev_report_sha256": "d",
    "test_sha256": "t",
    "model_digest": "m",
    "layer_fingerprint": "f",
}


def evaluate(tmp_path, test, **agent_kw):
    pool = cases(200, "p", 1)
    return runner.evaluate(
        PREREG,
        test,
        pool,
        pool,
        cases(100, "h", 2),
        FakeLayer(),
        arms_for(test, **agent_kw),
        DEV_REPORT,
        AUDIT,
        tmp_path,
        false_alarm_reps=2,
        log=lambda _: None,
    )


def test_resolve_scorer_accepts_only_plain_code_scorers():
    assert runner.resolve_scorer("agent_committee.scorers:exact_label")("a", "a")
    with pytest.raises(ValueError):
        runner.resolve_scorer("os:system")


def test_evaluate_runs_every_check_and_writes_predictions(tmp_path):
    test = cases(300, "t", 3)
    v = evaluate(tmp_path, test)
    names = {c.name for c in v.checks}
    expected = {"leakage", "power_cap", "quality_floor", "null_rate", "score_drift", "domain_auc"}
    assert expected <= names
    assert {"label_mix", "false_alarm_rate", "per_label_bias", "flagged_split"} <= names
    assert {"dev_gap[linear]", "dev_gap[agent]"} <= names
    lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 * 300
    assert v.audit.predictions_sha256 == runner.sha256_file(tmp_path / "predictions.jsonl")
    assert v.agent_arm == "agent" and v.simpler_arm == "linear"
    assert [a.name for a in v.arms] == ["agent", "linear", "prompt"]


def test_small_runs_cannot_pick_a_winner(tmp_path):
    v = evaluate(tmp_path, cases(300, "t", 3))
    assert v.verdict == Verdict.NEED_MORE_DATA
    assert any("power_cap" in r for r in v.reasons)


def test_copies_of_dev_are_dropped_before_the_run(tmp_path):
    test = cases(300, "t", 3)
    pool = cases(200, "p", 1)
    copied = [Case(id=f"x{i}", text=c.text, label=c.label) for i, c in enumerate(pool[:10])]
    v = evaluate(tmp_path, test + copied)
    assert v.n_dropped >= 10 and v.n_cases == 310 - v.n_dropped


def test_prereg_without_a_scorer_is_invalid_before_any_arm_runs(tmp_path):
    test = cases(50, "t", 3)
    pool = cases(50, "p", 1)
    bare = PREREG.model_copy(update={"scorer": None})
    v = runner.evaluate(
        bare,
        test,
        pool,
        pool,
        pool,
        FakeLayer(),
        {},
        DEV_REPORT,
        AUDIT,
        tmp_path,
        2,
        lambda _: None,
    )
    assert v.verdict == Verdict.INVALID and v.audit.predictions_sha256 is None
    assert not (tmp_path / "predictions.jsonl").exists()


def test_any_trace_of_a_sealed_run_blocks_another(tmp_path):
    assert runner.sealed_run_traces(tmp_path) == []
    for name in ("verdict.json", "predictions.jsonl", runner.RUN_MARKER):
        (tmp_path / name).write_text("x\n")
        assert tmp_path / name in runner.sealed_run_traces(tmp_path)


def test_main_checks_for_traces_before_it_unseals():
    source = open(runner.__file__, encoding="utf-8").read()
    assert source.index("sealed_run_traces(out_dir)") < source.index("seal.load_test(repo")


def test_agent_that_skips_its_tools_is_invalid(tmp_path):
    v = evaluate(tmp_path, cases(300, "t", 3), own_tools=False)
    assert v.verdict == Verdict.INVALID
    assert v.quality_floor_passed is False


def test_importing_the_runner_does_not_import_the_seal():
    import subprocess
    import sys

    code = "import sys, agent_committee.runner; print('agent_committee.seal' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "False"


def test_only_the_sealed_branch_imports_the_seal():
    """The one `seal` import sits in the else-branch of `if args.rehearse`, nowhere else."""
    import ast

    tree = ast.parse(open(runner.__file__, encoding="utf-8").read())

    def imports_seal(n):
        if isinstance(n, ast.Import):
            return any(a.name.endswith(".seal") or a.name == "seal" for a in n.names)
        if isinstance(n, ast.ImportFrom):
            return (n.module or "").endswith("seal") or any(a.name == "seal" for a in n.names)
        return False

    branches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "args.rehearse"
    ]
    assert len(branches) == 2  # choosing the cases, then the rehearsal note
    choose = branches[0]
    in_else = {id(n) for b in choose.orelse for n in ast.walk(b) if imports_seal(n)}
    everywhere = {id(n) for n in ast.walk(tree) if imports_seal(n)}
    assert in_else and everywhere == in_else


# ---------- v2 P3 ladder ----------
_ROOT = Path(__file__).resolve().parents[1]
_AN, _AP, _json = ArmName, ArmPrediction, json


def _pred(case_id, label, arm=_AN.LINEAR, pt=0, ct=0, lat=0.0):
    return _AP(
        case_id=case_id, arm=arm, pred=label, prompt_tokens=pt, completion_tokens=ct, latency_s=lat
    )


def test_runner_and_dev_evaluation_know_the_same_arms():
    assert set(runner.RUNNABLE) == set(_SIMPLER) | {"agent"}


def test_runner_runs_exactly_the_arms_of_the_dev_report():
    v1 = _json.loads(
        (_ROOT / "src/agent_committee/arms/dev_report.json").read_text(encoding="utf-8")
    )
    assert runner.arms_to_run(v1["results"]) == ["linear", "prompt", "agent"]  # v1 unchanged
    v2 = {n: {} for n in ("majority", "linear", "naive_bayes", "prompt", "agent", "cascade")}
    assert runner.arms_to_run(v2) == ["majority", "linear", "naive_bayes", "prompt", "agent"]
    tiers = {n: {} for n in ("boosting", "agent", "linear_svm", "random_forest", "linear")}
    assert runner.arms_to_run(tiers) == [
        "linear",
        "linear_svm",
        "random_forest",
        "boosting",
        "agent",
    ]
    assert runner.arms_to_run({"linear": {}, "prompt": {}}) is None  # no agent
    assert runner.arms_to_run({"agent": {}, "cascade": {}}) is None  # cascade without linear
    assert runner.arms_to_run({"agent": {}, "gpt_judge": {}}) is None  # unknown arm


class Fixed:
    def __init__(self, name, preds):
        self.name, self._preds = name, iter(preds)

    def predict(self, case_id, text):
        return next(self._preds)


def test_run_arms_derives_the_cascade_only_when_asked():
    cases = [Case(id=str(i), text="t", label="a") for i in range(3)]

    def arms():
        return {
            "linear": Fixed("linear", [_pred(str(i), lab) for i, lab in enumerate("aab")]),
            "agent": Fixed(
                "agent",
                [_pred(str(i), lab, _AN.AGENT, 10, 1, 0.5) for i, lab in enumerate("bba")],
            ),
        }

    plain = runner.run_arms(arms(), cases, lambda p, g: p == g, log=lambda _m: None)
    assert set(plain.runs) == {"linear", "agent"}
    with_cascade = runner.run_arms(
        arms(), cases, lambda p, g: p == g, log=lambda _m: None, cascade_flags=[False, True, True]
    )
    run = with_cascade.runs["cascade"]
    assert run.correct.tolist() == [True, False, True]
    assert np.array_equal(run.prompt_tokens, [0, 10, 10])
    assert sum(r["arm"] == "cascade" for r in with_cascade.rows) == 3
    assert with_cascade.own_tool_rate == plain.own_tool_rate  # the floor reads the agent only
