import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_app import H, make_app, wait_done
from test_app import root as _root_fixture

from verifier_server import flow

DATA = {
    "datasets": [
        {"option": "banking77", "status": "built in"},
        {"option": "emotion", "status": "available"},
    ],
    "labels": {"banking77": ["a", "b", "c"]},
    "plan_sizes": {"0.10": 168},
    "seconds_per_case": {"agent": 1.05},
    "price_tables": [],
    "saved_run": None,
}
root = _root_fixture  # the same temporary project folder as test_app

OK = {"scenario": "banking77", "gap": 0.1, "arms": ["agent", "linear"], "scope": ["a"]}


def test_arms_match_the_runners_runnable_list():
    from agent_committee.runner import RUNNABLE  # tests may import it; the server never does

    assert flow.ARMS == tuple(RUNNABLE)


def test_a_checked_choice_becomes_list_arguments_with_scope():
    cmd = flow.quick_command(OK, DATA)
    assert cmd[1:4] == ["-u", "-m", "agent_committee.quick"]
    assert cmd[cmd.index("--gap") + 1] == "0.10"
    assert cmd[cmd.index("--arms") + 1] == "linear,agent"
    assert cmd[-2:] == ["--scope", "a"]


def test_every_label_means_the_full_test_without_scope():
    assert "--scope" not in flow.quick_command({**OK, "scope": ["c", "a", "b"]}, DATA)


@pytest.mark.parametrize(
    "bad",
    [
        {"scenario": "emotion"},
        {"scenario": "../x"},
        {"gap": 0.07},
        {"gap": "0.1"},
        {"arms": ["linear"]},
        {"arms": ["agent", "prompt"]},
        {"arms": ["agent", "linear; rm -rf"]},
        {"scope": []},
        {"scope": ["zzz"]},
        {"scope": "a"},
    ],
)
def test_choices_outside_the_allow_lists_are_refused(bad):
    with pytest.raises(flow.FlowRefused):
        flow.quick_command({**OK, **bad}, DATA)


def test_the_users_agent_is_passed_with_its_price():
    cmd = flow.quick_command(
        {**OK, "agent": {"url": "http://127.0.0.1:8091/c", "usd_per_message": 0.002}}, DATA
    )
    assert cmd[cmd.index("--agent-url") + 1] == "http://127.0.0.1:8091/c"
    assert float(cmd[cmd.index("--agent-cost") + 1]) == 0.002
    assert "--agent-url" not in flow.quick_command(OK, DATA)


@pytest.mark.parametrize(
    "agent",
    [
        {"url": "file:///etc/passwd", "usd_per_message": 0},
        {"url": "http://x/c"},
        {"url": "http://x/c", "usd_per_message": -1},
        {"url": "http://x/c", "usd_per_message": 50},
        {"url": "http://x/c", "usd_per_message": True},
        "http://x/c",
    ],
)
def test_bad_agent_choices_are_refused(agent):
    with pytest.raises(flow.FlowRefused):
        flow.quick_command({**OK, "agent": agent}, DATA)


def test_the_single_prompt_model_is_refused_with_the_users_agent():
    with pytest.raises(flow.FlowRefused):
        flow.quick_command(
            {
                **OK,
                "arms": ["agent", "linear", "prompt"],
                "agent": {"url": "http://x/c", "usd_per_message": 0},
            },
            DATA,
        )


LINES = [
    "gap 0.100: 168 paired cases planned, 168 drawn from 189 DEV eval cases",
    "training linear ...",
    "trained linear in 0 s",
    "training random_forest ...",
    "trained random_forest in 41 s",
    "simpler model to beat (best free model on tune): linear {'linear': 0.98}",
    "  scored 30 of 168 cases",
    "  look at 34 cases: gain -0.088, z(agent ahead) -1.79, z(gain ruled out) -3.84, boundary 4.38",
    "  look at 67 cases: gain -0.119, z(agent ahead) -2.75, z(gain ruled out) -6.04, boundary 3.10",
    "stopped at 67 of 168 cases: simpler",
    "verdict: USE_SIMPLER -> results/quick/x/verdict.json",
]


def test_progress_is_read_from_the_quick_tests_own_lines():
    p: dict = {}
    for line in LINES:
        flow.read_progress(p, line)
    assert p["planned"] == 168 and p["simpler"] == "linear"
    assert p["trained"] == ["linear", "random_forest"] and "training" not in p
    assert p["scored"] == 30
    assert [look["n"] for look in p["looks"]] == [34, 67]
    assert p["looks"][1] == {
        "n": 67,
        "gain": -0.119,
        "z_agent": -2.75,
        "z_ruled_out": -6.04,
        "bound": 3.10,
    }
    assert p["stopped_at"] == 67 and p["verdict"] == "USE_SIMPLER" and p["phase"] == "done"
    assert p["result"] == "results/quick/x/verdict.json"


def write_run(root: Path, name: str = "x", scope: bool = True) -> Path:
    d = root / "results" / "quick" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps({"planned_cases": 168}), encoding="utf-8")
    (d / "verdict.json").write_text(json.dumps({"verdict": "USE_SIMPLER"}), encoding="utf-8")
    if scope:
        (d / "scope.json").write_text(json.dumps({"agent": {}}), encoding="utf-8")
    return d


def test_results_are_read_only_from_inside_results(root, tmp_path):
    write_run(root)
    got = flow.read_result(root, "results/quick/x/verdict.json")
    assert got["plan"]["planned_cases"] == 168 and got["scope"] == {"agent": {}}
    write_run(root, "y", scope=False)
    assert flow.read_result(root, "results/quick/y/verdict.json")["scope"] is None
    with pytest.raises(flow.FlowRefused):
        flow.read_result(root, "prereg.yaml")
    with pytest.raises(flow.FlowRefused):
        flow.read_result(root, "results/../verdict.json")


def quick_client(
    root: Path, monkeypatch, ollama: bool = True, lines: list[str] = LINES
) -> TestClient:
    monkeypatch.setattr(flow, "flow_data", lambda _root: DATA)
    script = "import time\n" + "".join(f"print({line!r}); time.sleep(0.02)\n" for line in lines)
    monkeypatch.setattr(flow, "quick_command", lambda choice, data: [sys.executable, "-c", script])
    return TestClient(make_app(root, ollama))


def test_flow_endpoint_lists_the_choices(root, monkeypatch):
    c = quick_client(root, monkeypatch)
    assert c.get("/api/flow").json()["labels"] == {"banking77": ["a", "b", "c"]}


def test_quick_needs_the_client_header_and_ollama(root, monkeypatch):
    assert quick_client(root, monkeypatch).post("/api/jobs/quick", json=OK).status_code == 403
    down = quick_client(root, monkeypatch, ollama=False)
    assert down.post("/api/jobs/quick", json=OK, headers=H).status_code == 503


def test_a_refused_choice_is_422(root, monkeypatch):
    monkeypatch.setattr(flow, "flow_data", lambda _root: DATA)
    c = TestClient(make_app(root))
    assert c.post("/api/jobs/quick", json={**OK, "gap": 0.33}, headers=H).status_code == 422


def test_quick_job_reports_progress_then_its_result(root, monkeypatch):
    write_run(root)
    c = quick_client(root, monkeypatch)
    job = c.post("/api/jobs/quick", json=OK, headers=H).json()
    assert job["kind"] == "quick"
    assert c.post("/api/jobs/quick", json=OK, headers=H).status_code == 409  # one job at a time
    done = wait_done(c, job["id"])
    assert done["state"] == "succeeded"
    assert len(done["progress"]["looks"]) == 2 and done["progress"]["phase"] == "done"
    assert c.get(f"/api/jobs/{job['id']}/result").json()["verdict"] == {"verdict": "USE_SIMPLER"}


def test_no_result_before_the_job_succeeds(root, monkeypatch):
    c = quick_client(root, monkeypatch, lines=["gap 0.100: 168 paired cases planned, 168 drawn"])
    job = c.post("/api/jobs/quick", json=OK, headers=H).json()
    wait_done(c, job["id"])
    assert c.get(f"/api/jobs/{job['id']}/result").status_code == 409
    assert c.get("/api/jobs/nope/result").status_code == 404


def test_training_shows_the_model_being_trained():
    p: dict = {}
    flow.read_progress(p, "training boosting ...")
    assert p == {"training": "boosting", "phase": "training"}


def test_neighbours_are_the_most_similar_messages():
    nb = flow.neighbours(["refund please", "refund pls", "card lost", "lost my card"], k=1)
    assert nb == [[1], [0], [3], [2]]


def test_run_messages_join_both_answers_in_the_agents_order(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import agent_committee.data as data
    import agent_committee.scenarios as scenarios

    cases = [
        SimpleNamespace(id="c1", text="refund please", label="refund"),
        SimpleNamespace(id="c2", text="card lost", label="lost_card"),
    ]
    monkeypatch.setattr(data, "load_dev", lambda _d: cases)
    monkeypatch.setattr(scenarios, "get_scenario", lambda _n, _r: SimpleNamespace(data_dir="d"))
    rows = [
        {"arm": "linear", "case_id": "c1", "pred": "refund", "correct": True},
        {"arm": "agent", "case_id": "c2", "pred": "refund", "correct": False, "own_tool_calls": 2},
        {"arm": "agent", "case_id": "c1", "pred": "refund", "correct": True, "latency_s": 1.5},
        {"arm": "linear", "case_id": "c2", "pred": "not_mine", "correct": True},
        {"arm": "majority", "case_id": "c1", "pred": "x", "correct": False},
    ]
    (tmp_path / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    plan = {"scenario": "banking77", "comparator": "linear", "scope": {"labels": ["refund"]}}
    got = flow.run_messages(tmp_path, tmp_path, plan)
    assert [m["text"] for m in got] == ["card lost", "refund please"]
    assert got[0]["gold"] == "not_mine" and not got[0]["agent_ok"] and got[0]["simpler_ok"]
    assert got[0]["agent_calls"] == 2 and got[1]["agent_latency"] == 1.5
    assert got[0]["nb"] == [1] and got[1]["nb"] == [0]


def test_library_endpoint_refuses_unknown_datasets(root, monkeypatch):
    monkeypatch.setattr(flow, "flow_data", lambda _root: DATA)
    monkeypatch.setattr(flow, "library", lambda _root, name: [{"text": "t", "label": "a"}])
    client = TestClient(make_app(root, True))
    got = client.get("/api/flow/library", params={"dataset": "banking77"}).json()
    assert got == [{"text": "t", "label": "a"}]
    assert client.get("/api/flow/library", params={"dataset": "emotion"}).status_code == 404


def _csv(rows):
    return ("text,label\n" + "".join(f"{t},{label}\n" for t, label in rows)).encode()


def test_own_ids_come_from_the_rows_not_their_order():
    rows = [("hello there", "greet"), ("bye now", "leave"), ("hello there", "greet")]
    a = {c.id for c in flow.own_cases(_csv(rows), "csv")}
    b = {c.id for c in flow.own_cases(_csv(rows[::-1]), "csv")}
    assert a == b and len(a) == 3


def test_own_rows_need_text_and_label():
    with pytest.raises(flow.FlowRefused, match="row 1"):
        flow.own_cases(b"text,label\nhi,\n", "csv")
    with pytest.raises(flow.FlowRefused):
        flow.own_cases(b"{}", "xlsx")


def test_detectable_gap_is_the_smallest_test_that_fits():
    sizes = {"0.05": 658, "0.10": 168, "0.20": 39}
    assert flow.detectable(200, sizes) == 0.10
    assert flow.detectable(700, sizes) == 0.05
    assert flow.detectable(20, sizes) is None


PHRASE = {"item": "a message", "items": "messages", "label_kind": "topic"}


def test_own_data_is_refused_before_anything_is_written(tmp_path):
    import shutil

    shutil.copy(Path(__file__).parents[2] / "prereg.yaml", tmp_path / "prereg.yaml")
    few = _csv([(f"note {i}", "a" if i % 2 else "b") for i in range(20)])
    with pytest.raises(flow.FlowRefused, match="split fairly"):
        flow.add_own(tmp_path, "mine", "csv", few, PHRASE)
    with pytest.raises(flow.FlowRefused, match="lowercase"):
        flow.add_own(tmp_path, "Bad Name", "csv", few, PHRASE)
    with pytest.raises(flow.FlowRefused, match="already exists"):
        flow.add_own(tmp_path, "banking77", "csv", few, PHRASE)
    assert not (tmp_path / "sources").exists() and not (tmp_path / "scenarios").exists()


def test_own_data_is_split_once_and_says_what_it_can_detect(tmp_path):
    import shutil

    shutil.copy(Path(__file__).parents[2] / "prereg.yaml", tmp_path / "prereg.yaml")
    rows = [
        (f"{w} request number {i}", w) for w in ("refund", "card", "transfer") for i in range(120)
    ]
    out = flow.add_own(tmp_path, "mine", "csv", _csv(rows), PHRASE, "My bank bot")
    assert out["cases"] == 360 and out["labels"] == 3 and out["eval"] > 0
    assert out["detectable_gap"] in (None, 0.2, 0.15, 0.1, 0.05, 0.02)
    assert (tmp_path / "data" / "mine" / "dev.jsonl").is_file()
    with pytest.raises(flow.FlowRefused, match="already exists"):
        flow.add_own(tmp_path, "mine", "csv", _csv(rows), PHRASE)


def test_an_added_dataset_is_listed_with_its_title(tmp_path, monkeypatch):
    import shutil

    from agent_committee import catalog

    shutil.copy(Path(__file__).parents[2] / "prereg.yaml", tmp_path / "prereg.yaml")
    rows = [
        (f"{w} request number {i}", w) for w in ("refund", "card", "transfer") for i in range(120)
    ]
    flow.add_own(tmp_path, "mine", "csv", _csv(rows), PHRASE, "My bank bot")
    monkeypatch.setattr(catalog, "menu", lambda _root: [])
    data = flow.flow_data(tmp_path)
    assert data["datasets"] == [
        {
            "option": "mine",
            "title": "My bank bot",
            "description": "",
            "status": "split built",
            "own": True,
        }
    ]
    assert (
        data["labels"]["mine"] == ["card", "refund", "transfer"] and data["eval_sizes"]["mine"] > 0
    )


def test_an_unreachable_agent_is_explained_in_plain_words():
    import urllib.error

    refused = urllib.error.URLError(ConnectionRefusedError(10061, "refused"))
    assert flow.unreachable(refused).startswith("Nothing is answering at that address")
    assert "15 seconds" in flow.unreachable(urllib.error.URLError(TimeoutError()))
    assert "JSON" in flow.unreachable(ValueError("bad"))


def test_same_place_needs_two_or_more_picked_kinds_not_all():
    two = {**OK, "scope": ["a", "b"], "one_route": True}
    assert flow.quick_command(two, DATA)[-1] == "--one-route"
    assert "--one-route" not in flow.quick_command({**OK, "scope": ["a", "b"]}, DATA)
    for bad in (
        {**OK, "one_route": True},
        {**OK, "scope": ["a", "b", "c"], "one_route": True},
        {**OK, "one_route": "yes"},
    ):
        with pytest.raises(flow.FlowRefused):
            flow.quick_command(bad, DATA)


@pytest.mark.parametrize("bad", [["banking77"], {"a": 1}, 7, None])
def test_wrongly_shaped_datasets_are_refused_not_crashed(bad):
    with pytest.raises(flow.FlowRefused):
        flow.quick_command({**OK, "scenario": bad}, DATA)
