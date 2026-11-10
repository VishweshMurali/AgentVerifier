"""MCP data endpoints: the flow works end to end and nothing ever exposes TEST."""

import ast
import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from fastmcp import Client
from fastmcp.exceptions import ToolError
from verifier_mcp.server import build_server

from agent_committee.data import build_split as build_split_cli
from agent_committee.scenarios import get_scenario

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "mcp_server" / "src" / "verifier_mcp"
PER_LABEL = 30


def call(server, tool, **args):
    async def go():
        async with Client(server) as client:
            result = await client.call_tool(tool, args)
            return result.data

    return asyncio.run(go())


def tool_names(server):
    async def go():
        async with Client(server) as client:
            return {t.name: t for t in await client.list_tools()}

    return asyncio.run(go())


@pytest.fixture
def repo(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    rows = [
        {"id": f"{x}{i:02d}", "text": f"unique case text {x} {i}", "label": x}
        for x in "abc"
        for i in range(PER_LABEL)
    ]
    (sources / "demo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    return tmp_path


def write_prereg(repo: Path, name: str = "demo") -> None:
    prereg = yaml.safe_load((REPO / "prereg.yaml").read_text(encoding="utf-8"))
    prereg["task"] = name
    path = repo / "scenarios" / name / "prereg.yaml"
    path.write_text(yaml.safe_dump(prereg), encoding="utf-8")


def test_full_flow_and_no_tool_output_contains_a_test_case(repo):
    server = build_server(repo)
    outputs = []
    out = call(server, "add_dataset", name="demo", kind="local", path="sources/demo.jsonl")
    assert out["written"] == "scenarios/demo/scenario.yaml"
    outputs.append(out)
    check = call(server, "check_dataset", name="demo")
    assert check["n_cases"] == 3 * PER_LABEL and check["problems"] == []
    assert check["prereg"].startswith("missing") and check["split_built"] is False
    assert check["build_command"] is None  # no prereg yet
    outputs.append(check)
    outputs.append(call(server, "prereg_template", name="demo"))
    write_prereg(repo)
    ready = call(server, "check_dataset", name="demo")
    assert ready["build_command"] == "uv run python -m agent_committee.data --scenario demo"
    outputs.append(ready)
    # the owner runs that command by hand (no endpoint can build a split)
    manifest = build_split_cli(get_scenario("demo", repo), repo)
    assert manifest["n_dev"] + manifest["n_test"] == 3 * PER_LABEL
    stats = call(server, "dev_stats", name="demo")
    assert stats["n_dev"] == manifest["n_dev"] and stats["manifest"]["n_test"] == manifest["n_test"]
    outputs.append(stats)
    listed = {s["name"]: s for s in call(server, "list_scenarios")}
    assert listed["demo"]["split_built"] and listed["demo"]["source"] == "local"
    assert listed["banking77"]["source"] == "built-in"
    outputs.append(listed)
    assert call(server, "check_dataset", name="demo")["build_command"] is None  # built already

    sealed = (repo / "sealed" / "demo" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    test_texts = [json.loads(line)["text"] for line in sealed]
    blob = json.dumps(outputs, default=str)
    assert test_texts and not [t for t in test_texts if t in blob]
    assert "agent_committee.seal" not in sys.modules and "agent_committee.runner" not in sys.modules


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ({"name": "demo", "kind": "local", "path": "../outside.jsonl"}, "inside sources/"),
        ({"name": "demo", "kind": "local", "path": "sources/x.txt"}, "jsonl or .csv"),
        ({"name": "../evil", "kind": "local", "path": "sources/demo.jsonl"}, "lowercase"),
        ({"name": "banking77", "kind": "local", "path": "sources/demo.jsonl"}, "menu option"),
        ({"name": "rehearsal", "kind": "local", "path": "sources/demo.jsonl"}, "reserved"),
        ({"name": "demo", "kind": "pickle", "path": "sources/demo.jsonl"}, "kind must be"),
    ],
)
def test_add_dataset_refuses_paths_names_and_kinds_outside_the_rules(repo, args, error):
    with pytest.raises(ToolError, match=error):
        call(build_server(repo), "add_dataset", **args)
    assert not (repo / "scenarios").exists() or not list((repo / "scenarios").rglob("*.yaml"))


def test_a_symlink_out_of_sources_is_refused(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    link = repo / "sources" / "link.jsonl"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks need extra rights on this system")
    with pytest.raises(ToolError, match="inside sources/"):
        call(
            build_server(repo), "add_dataset", name="demo", kind="local", path="sources/link.jsonl"
        )


def test_huggingface_dataset_is_declared_without_downloading(repo, monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)  # any import of datasets would fail
    out = call(
        build_server(repo),
        "add_dataset",
        name="clinc",
        kind="huggingface",
        dataset="clinc_oos",
        config="plus",
        text="text",
        label="intent",
        splits=["train", "validation", "test"],
    )
    spec = yaml.safe_load((repo / out["written"]).read_text(encoding="utf-8"))
    assert spec["source"] == {
        "kind": "huggingface",
        "dataset": "clinc_oos",
        "config": "plus",
        "text": "text",
        "label": "intent",
        "splits": ["train", "validation", "test"],
    }


def test_no_endpoint_builds_a_split_and_http_default_is_read_only(repo):
    assert set(tool_names(build_server(repo, allow_writes=False))) == {
        "list_datasets",
        "list_scenarios",
        "check_dataset",
        "prereg_template",
        "dev_stats",
    }
    tools = tool_names(build_server(repo))
    read_only = {"list_datasets", "list_scenarios", "check_dataset", "prereg_template", "dev_stats"}
    assert set(tools) == {*read_only, "add_dataset", "choose_dataset"}
    assert tools["list_scenarios"].annotations.read_only_hint is True


def test_prereg_template_leaves_the_owners_values_as_todos(repo):
    shutil.copy(REPO / "prereg.yaml", repo / "prereg.yaml")
    text = call(build_server(repo), "prereg_template", name="banking77")
    assert "task: banking77" in text and "min_gap: TODO" in text and "low: TODO" in text
    assert not (repo / "scenarios").exists()


def test_the_package_never_imports_the_seal_or_runner_or_names_the_test_file():
    for path in PKG.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "test.jsonl" not in source, path.name
        for node in ast.walk(ast.parse(source)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)]
            for n in names:
                assert not n.endswith((".seal", ".runner")), f"{path.name}: {n}"


def test_a_half_filled_prereg_and_a_broken_scenario_file_are_reported_not_raised(repo):
    server = build_server(repo)
    call(server, "add_dataset", name="demo", kind="local", path="sources/demo.jsonl")
    template = call(server, "prereg_template", name="demo")
    assert "# linear: filled from THIS dataset" in template and "linear: TODO" not in template
    (repo / "scenarios" / "demo" / "prereg.yaml").write_text(template, encoding="utf-8")
    check = call(server, "check_dataset", name="demo")
    assert check["prereg"].startswith("invalid, split defaults used") and check["n_cases"] == 90
    broken = repo / "scenarios" / "bad" / "scenario.yaml"
    broken.parent.mkdir()
    broken.write_text("name: bad\nsource: {kind: ftp}\n", encoding="utf-8")
    listed = {s["name"]: s for s in call(server, "list_scenarios")}
    assert "error" in listed["bad"] and listed["demo"]["source"] == "local"


def test_a_yaml_syntax_error_in_one_scenario_file_does_not_hide_the_rest(repo):
    broken = repo / "scenarios" / "bad" / "scenario.yaml"
    broken.parent.mkdir(parents=True)
    broken.write_text("name: bad\nsource: {kind: local, path: [unclosed\n", encoding="utf-8")
    listed = {s["name"]: s for s in call(build_server(repo), "list_scenarios")}
    assert "error" in listed["bad"] and listed["banking77"]["source"] == "built-in"


# ---------- the dataset menu ----------
def test_list_datasets_shows_the_menu(repo):
    options = [m["option"] for m in call(build_server(repo), "list_datasets")]
    assert options == ["banking77", "clinc150", "emotion", "own"]


def test_choose_dataset_by_option(repo):
    out = call(build_server(repo), "choose_dataset", option="emotion")
    assert out["status"] == "chosen" and (repo / "scenarios/emotion/scenario.yaml").is_file()


def test_choose_dataset_without_a_pick_list_returns_the_menu(repo):
    out = call(build_server(repo), "choose_dataset")
    assert [m["option"] for m in out["pick_one"]][:3] == ["banking77", "clinc150", "emotion"]
    assert not (repo / "scenarios").exists()


def test_add_dataset_cannot_take_a_menu_options_name(repo):
    with pytest.raises(ToolError, match="use choose_dataset"):
        call(
            build_server(repo),
            "add_dataset",
            name="clinc150",
            kind="local",
            path="sources/demo.jsonl",
        )


def test_preload_is_refused_on_stdio():
    from verifier_mcp.__main__ import main

    with pytest.raises(SystemExit):
        main(["--preload"])
