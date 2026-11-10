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
    outputs.append(check)
    outputs.append(call(server, "prereg_template", name="demo"))
    write_prereg(repo)
    with pytest.raises(ToolError, match="irreversible"):
        call(server, "build_split", name="demo")
    manifest = call(server, "build_split", name="demo", confirm=True)
    assert manifest["n_dev"] + manifest["n_test"] == 3 * PER_LABEL
    outputs.append(manifest)
    stats = call(server, "dev_stats", name="demo")
    assert stats["n_dev"] == manifest["n_dev"] and stats["manifest"]["n_test"] == manifest["n_test"]
    outputs.append(stats)
    listed = {s["name"]: s for s in call(server, "list_scenarios")}
    assert listed["demo"]["split_built"] and listed["demo"]["source"] == "local"
    assert listed["banking77"]["source"] == "built-in"
    outputs.append(listed)
    with pytest.raises(ToolError, match="already exists"):
        call(server, "build_split", name="demo", confirm=True)

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
        ({"name": "banking77", "kind": "local", "path": "sources/demo.jsonl"}, "built-in"),
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


def test_read_only_server_has_no_write_tools_and_split_is_marked_destructive(repo):
    assert set(tool_names(build_server(repo, allow_writes=False))) == {
        "list_scenarios",
        "check_dataset",
        "prereg_template",
        "dev_stats",
    }
    tools = tool_names(build_server(repo))
    assert tools["build_split"].annotations.destructive_hint is True
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
    assert "linear: TODO" in template and "never" in template
    (repo / "scenarios" / "demo" / "prereg.yaml").write_text(template, encoding="utf-8")
    check = call(server, "check_dataset", name="demo")
    assert check["prereg"].startswith("invalid, split defaults used") and check["n_cases"] == 90
    broken = repo / "scenarios" / "bad" / "scenario.yaml"
    broken.parent.mkdir()
    broken.write_text("name: bad\nsource: {kind: ftp}\n", encoding="utf-8")
    listed = {s["name"]: s for s in call(server, "list_scenarios")}
    assert "error" in listed["bad"] and listed["demo"]["source"] == "local"
