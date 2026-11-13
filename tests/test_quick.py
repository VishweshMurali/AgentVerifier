import dataclasses
import hashlib
import json
import zlib
from pathlib import Path

import pytest

from agent_committee import model_cache, power, quick
from agent_committee.schemas import ArmName, ArmPrediction, Case


@pytest.fixture(autouse=True)
def _own_model_cache(tmp_path, monkeypatch):
    """Saved models from tests go to a temporary folder, never the project's cache."""
    monkeypatch.setattr(model_cache, "CACHE_DIR", tmp_path / "saved-models")


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
    monkeypatch.setattr(model_cache, "CACHE_DIR", tmp_path / "models")  # never the real cache
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


class ShortlistLayer:
    """decide(text) shortlists by the case's own text: 'a ...' -> [a, x], 'b ...' -> [b, y, x]."""

    def decide(self, text):
        from types import SimpleNamespace

        first = text.split()[0]
        return SimpleNamespace(
            shortlist={"a": ["a", "x"], "b": ["b", "y", "x"]}.get(first, [first])
        )


def test_near_misses_are_the_out_of_scope_labels_shortlisted_most_for_scope_cases():
    pool = [Case(id=f"p{i}", text=f"{lb} t", label=lb) for i, lb in enumerate("aabbz")]
    assert quick.near_misses(ShortlistLayer(), pool, ["a", "b"]) == ["x", "y"]  # x 4, y 2
    assert quick.near_misses(ShortlistLayer(), pool, ["a"]) == ["x"]  # at most len(scope)


def test_scoped_mix_keeps_scope_and_near_misses_and_a_fixed_slice_of_others():
    cases = [Case(id=f"c{i:03d}", text="t", label=lb)
             for i, lb in enumerate(["in"] * 30 + ["near"] * 10 + ["far"] * 100)]  # fmt: skip
    mix = quick.scoped(cases, ["in"], ["near"])
    labels = [c.label for c in mix]
    assert labels.count("in") == 30 and labels.count(quick.NOT_MINE) == 10 + 10  # 10 = 40 * 0.25
    assert mix == quick.scoped(list(reversed(cases)), ["in"], ["near"])
    assert all(c.label in ("in", quick.NOT_MINE) for c in mix)


def test_in_scope_maps_other_answers_to_not_mine_and_keeps_no_answer():
    class Answers:
        name = "x"

        def __init__(self, pred):
            self.pred = pred

        def predict(self, case_id, text):
            return ArmPrediction(case_id=case_id, arm=ArmName.LINEAR, pred=self.pred)

    assert quick.in_scope(Answers("far"), ["in"]).predict("c", "t").pred == quick.NOT_MINE
    assert quick.in_scope(Answers("in"), ["in"]).predict("c", "t").pred == "in"
    assert quick.in_scope(Answers(None), ["in"]).predict("c", "t").pred is None
    raw = Answers("far")
    assert quick.in_scope(raw, []) is raw


def test_scope_summary_counts_in_scope_accuracy_and_false_grabs(tmp_path):
    used = [Case(id="1", text="t", label="in"), Case(id="2", text="t", label=quick.NOT_MINE)]
    rows = [
        {"arm": "agent", "case_id": "1", "pred": "in", "correct": True},
        {"arm": "agent", "case_id": "2", "pred": "in", "correct": False},
    ]
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    s = quick.scope_summary(path, used, ["in"])["agent"]
    assert s["in_scope_accuracy"] == 1.0 and s["false_grab_rate"] == 1.0


def test_unknown_scope_labels_are_refused(capsys):
    assert quick.main(["--scenario", "banking77", "--gap", "0.1", "--scope", "card_arrival,nope",
                       "--arms", "linear,agent", "--yes"]) == 1  # fmt: skip
    assert "nope" in capsys.readouterr().err


def test_scoped_end_to_end_on_banking77_card_labels(tmp_path, monkeypatch):
    _fake_run(tmp_path, monkeypatch)
    scope = "card_arrival,card_delivery_estimate,activate_my_card,lost_or_stolen_card"
    code = quick.main(["--scenario", "banking77", "--gap", "0.15", "--scope", scope,
                       "--arms", "linear,agent", "--yes"])  # fmt: skip
    assert code == 0
    [run] = list((tmp_path / "quick").iterdir())
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    assert plan["scope"]["labels"] == sorted(scope.split(","))
    assert len(plan["scope"]["near_misses"]) == 4
    summary = json.loads((run / "scope.json").read_text(encoding="utf-8"))
    assert set(summary) >= {"linear", "agent"}
    verdict = json.loads((run / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] != "INVALID"  # the scoped mix is not mistaken for drift
    assert any(r.startswith("Scope: 4 labels") for r in verdict["reasons"])


def test_one_route_scores_a_mix_up_between_picked_kinds_as_right():
    cases = [Case(id=f"c{i:03d}", text="t", label=lb)
             for i, lb in enumerate(["a"] * 10 + ["b"] * 10 + ["far"] * 100)]  # fmt: skip
    mix = quick.scoped(cases, ["a", "b"], [], route=True)
    assert {c.label for c in mix} == {quick.ROUTE, quick.NOT_MINE}

    class Answers:
        name = "x"

        def __init__(self, pred):
            self.pred = pred

        def predict(self, case_id, text):
            return ArmPrediction(case_id=case_id, arm=ArmName.LINEAR, pred=self.pred)

    # "b" for an "a" message is the same route; strict scoring keeps the kind itself
    assert (
        quick.in_scope(Answers("b"), ["a", "b"], route=True).predict("c", "t").pred == quick.ROUTE
    )
    assert quick.in_scope(Answers("b"), ["a", "b"]).predict("c", "t").pred == "b"
    assert (
        quick.in_scope(Answers("far"), ["a", "b"], route=True).predict("c", "t").pred
        == quick.NOT_MINE
    )
    assert quick.in_scope(Answers(None), ["a", "b"], route=True).predict("c", "t").pred is None


def test_scope_summary_reads_the_route_label(tmp_path):
    used = [Case(id="1", text="t", label=quick.ROUTE), Case(id="2", text="t", label=quick.NOT_MINE)]
    rows = [
        {"arm": "agent", "case_id": "1", "pred": quick.ROUTE, "correct": True},
        {"arm": "agent", "case_id": "2", "pred": quick.ROUTE, "correct": False},
    ]
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    s = quick.scope_summary(path, used, ["a", "b"], route=True)["agent"]
    assert s["in_scope_cases"] == 1 and s["in_scope_accuracy"] == 1.0
    assert s["other_cases"] == 1 and s["false_grab_rate"] == 1.0


@pytest.mark.parametrize("scope", [None, "card_arrival"])
def test_one_route_needs_two_or_more_picked_kinds(scope, capsys):
    args = [
        "--scenario",
        "banking77",
        "--gap",
        "0.1",
        "--arms",
        "linear,agent",
        "--yes",
        "--one-route",
    ]
    if scope:
        args += ["--scope", scope]
    assert quick.main(args) == 1
    assert "two or more kinds" in capsys.readouterr().err


def test_one_route_end_to_end_gets_its_own_folder_and_says_so(tmp_path, monkeypatch):
    _fake_run(tmp_path, monkeypatch)
    scope = "card_arrival,card_delivery_estimate"
    base = [
        "--scenario",
        "banking77",
        "--gap",
        "0.2",
        "--scope",
        scope,
        "--arms",
        "linear,agent",
        "--yes",
    ]
    assert quick.main(base) == 0
    assert quick.main([*base, "--one-route"]) == 0
    runs = sorted(p.name for p in (tmp_path / "quick").iterdir())
    assert len(runs) == 2 and "-route-" in runs[1] and "-route-" not in runs[0]
    route_run = tmp_path / "quick" / runs[1]
    plan = json.loads((route_run / "plan.json").read_text(encoding="utf-8"))
    assert plan["scope"]["one_route"] is True and plan["scope"]["route_label"] == quick.ROUTE
    verdict = json.loads((route_run / "verdict.json").read_text(encoding="utf-8"))
    assert any(r.startswith("One route: the 2 picked kinds") for r in verdict["reasons"])


def test_saved_models_are_reused_only_for_identical_data_and_code(tmp_path):
    fits = []

    def fit():
        fits.append(1)
        return {"model": len(fits)}

    key = model_cache.model_key("m", "data1", "code1")
    first, _, saved1 = model_cache.load_or_fit(tmp_path, "m", key, fit)
    again, _, saved2 = model_cache.load_or_fit(tmp_path, "m", key, fit)
    assert (saved1, saved2) == (False, True) and again == first and len(fits) == 1
    other = model_cache.model_key("m", "data2", "code1")
    assert other != key and model_cache.load_or_fit(tmp_path, "m", other, fit)[2] is False
    # a broken file is refitted, not trusted
    (tmp_path / model_cache.CACHE_DIR / f"m-{key}.pkl").write_bytes(b"not a pickle")
    assert model_cache.load_or_fit(tmp_path, "m", key, fit)[2] is False


def test_data_key_follows_the_messages_not_the_order():
    a = [Case(id="1", text="x", label="p"), Case(id="2", text="y", label="q")]
    assert model_cache.data_digest(a) == model_cache.data_digest(list(reversed(a)))
    changed = [a[0], Case(id="2", text="y", label="r")]
    assert model_cache.data_digest(changed) != model_cache.data_digest(a)


def test_a_saved_model_predicts_exactly_like_a_fresh_fit(tmp_path):
    from agent_committee.arms.classic import NaiveBayesArm
    from agent_committee.data import load_dev

    pool = load_dev(Path("data"))[:600]

    def fit():
        arm = NaiveBayesArm()
        arm.fit(pool)
        return arm

    key = model_cache.model_key("naive_bayes", model_cache.data_digest(pool), "c")
    fresh, _, _ = model_cache.load_or_fit(tmp_path, "naive_bayes", key, fit)
    loaded, _, saved = model_cache.load_or_fit(tmp_path, "naive_bayes", key, fit)
    assert saved
    for c in pool[:200]:
        assert (
            loaded.predict(c.id, c.text).pred
            == fresh.predict(c.id, c.text).pred
            == fit().predict(c.id, c.text).pred
        )


def test_each_case_is_answered_once_and_the_verdict_uses_the_stopping_answers(
    tmp_path, monkeypatch
):
    """R12: the agent was asked again after stopping (16 calls for 8 cases), and the verdict used
    the second answers. A changing agent must be called once per case and judged on those."""
    from agent_committee.arms import agent as agent_arm
    from agent_committee.data import load_dev

    _fake_run(tmp_path, monkeypatch)
    labels = {c.id: c.label for c in load_dev(Path("data"))}
    calls: dict[str, int] = {}

    class Changing:
        name = ArmName.AGENT

        def predict(self, case_id, text):
            calls[case_id] = calls.get(case_id, 0) + 1
            # right the first time it is asked, wrong any later time
            pred = labels[case_id] if calls[case_id] == 1 else "wrong"
            return ArmPrediction(case_id=case_id, arm=self.name, pred=pred, own_tool_calls=1)

    monkeypatch.setattr(agent_arm, "make_arm", lambda *a, **k: Changing())
    assert (
        quick.main(["--scenario", "banking77", "--gap", "0.2", "--arms", "linear,agent", "--yes"])
        == 0
    )
    assert calls and set(calls.values()) == {1}
    [run] = list((tmp_path / "quick").iterdir())
    rows = [
        json.loads(x) for x in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    agent_rows = [r for r in rows if r["arm"] == "agent"]
    assert len(agent_rows) == len(calls) and all(r["correct"] for r in agent_rows)


def test_your_agent_runs_without_ollama(tmp_path, monkeypatch):
    """R12: an external-only test built the Ollama client first and stopped when it was off."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from agent_committee import llm

    _fake_run(tmp_path, monkeypatch)

    class NoOllama:
        def __init__(self, **kw):
            raise ConnectionError("Ollama is off")

    monkeypatch.setattr(llm, "LLMClient", NoOllama)

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            out = json.dumps({"label": body["labels"][0]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/c"
        args = ["--scenario", "banking77", "--gap", "0.2", "--arms", "linear,agent", "--yes",
                "--agent-url", url, "--agent-cost", "0.001"]  # fmt: skip
        assert quick.main(args) == 0
        # the single-prompt model does need Ollama, so it still refuses when Ollama is off
        with pytest.raises(ConnectionError):
            quick.main([a if a != "linear,agent" else "linear,prompt,agent" for a in args])
    finally:
        srv.shutdown()


def test_every_run_gets_its_own_folder(tmp_path):
    a = quick.new_run_dir(tmp_path, "gap0.100-n90")
    b = quick.new_run_dir(tmp_path, "gap0.100-n90")
    assert a != b and a.is_dir() and b.is_dir() and a.name.startswith("gap0.100-n90-")


def test_training_key_cannot_be_forged_by_separator_characters():
    one = [Case(id="1", text="a\x1fb", label="c")]
    two = [Case(id="1", text="a", label="b\x1fc")]
    assert model_cache.data_digest(one) != model_cache.data_digest(two)
    three = [Case(id="1", text="x\n2\x1fy", label="z")]
    four = [Case(id="1", text="x", label="z"), Case(id="2", text="y", label="z")]
    assert model_cache.data_digest(three) != model_cache.data_digest(four)
