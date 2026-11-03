"""Static guard: arm code may only see DEV data and may not reach the sealed TEST set."""

import ast
from pathlib import Path

ARMS = Path(__file__).resolve().parents[1] / "src" / "agent_committee" / "arms"
ALLOWED_TOP = {
    "__future__", "argparse", "collections", "dataclasses", "enum", "functools", "hashlib",
    "itertools", "json", "logging", "math", "numpy", "pathlib", "pydantic", "re", "scipy",
    "sklearn", "sys", "textwrap", "typing", "langgraph",
}  # fmt: skip
ALLOWED_PROJECT = {
    "agent_committee.schemas",
    "agent_committee.llm",
    "agent_committee.data",
    "agent_committee.scorers",
}
BANNED_TEXT = [
    "seal", "importlib", "subprocess", "test.jsonl", "test.sha256", "SEALED_TEST_FILE",
    "__import__", "pickle",
]  # fmt: skip


def _files():
    return sorted(ARMS.rglob("*.py"))


def test_no_runner_in_arms():
    assert not [p for p in ARMS.rglob("*") if p.name == "runner.py"]


def test_imports_are_whitelisted():
    for path in _files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name.split(".")[0] in ALLOWED_TOP or a.name.startswith(
                        "agent_committee.arms"
                    ), f"{path.name}: import {a.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    continue  # relative import inside arms/
                if mod.split(".")[0] in ALLOWED_TOP or mod.startswith("agent_committee.arms"):
                    continue
                if mod == "agent_committee":
                    names = {a.name for a in node.names}
                    assert names <= {"schemas", "llm"}, f"{path.name}: from agent_committee {names}"
                    continue
                assert mod in ALLOWED_PROJECT, f"{path.name}: from {mod}"
                if mod == "agent_committee.data":
                    assert {a.name for a in node.names} == {"load_dev"}, (
                        f"{path.name}: only data.load_dev may be imported"
                    )


def test_no_banned_text():
    for path in _files():
        src = path.read_text(encoding="utf-8")
        for word in BANNED_TEXT:
            assert word not in src, f"{path.name} contains {word!r}"


def test_no_clock_or_unseeded_randomness():
    for path in _files():
        src = path.read_text(encoding="utf-8")
        for word in ("import time", "import datetime", "import random", "uuid", "np.random."):
            assert word not in src, f"{path.name} contains {word!r}"
