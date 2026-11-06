"""Static guard: arm code may only see DEV data and may not reach the sealed TEST set."""

import ast
from pathlib import Path

import pytest

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


def _numpy_random_uses(source: str) -> list[str]:
    """Every way `source` reaches numpy.random: direct imports, `from` imports and aliases."""
    tree = ast.parse(source)
    numpy_names = set()  # local names bound to the numpy module
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "numpy.random" or a.name.startswith("numpy.random."):
                    hits.append(f"import {a.name}")
                elif a.name == "numpy":
                    numpy_names.add(a.asname or "numpy")
        elif isinstance(node, ast.ImportFrom) and not node.level:
            mod = node.module or ""
            if mod == "numpy.random" or mod.startswith("numpy.random."):
                hits.append(f"from {mod} import ...")
            elif mod == "numpy" and any(a.name == "random" for a in node.names):
                hits.append("from numpy import random")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "random"
            and isinstance(node.value, ast.Name)
            and node.value.id in numpy_names
        ):
            hits.append(f"{node.value.id}.random")
    return hits


@pytest.mark.parametrize(
    "source",
    [
        "from numpy.random import default_rng",
        "from numpy import random",
        "from numpy import random as r",
        "import numpy.random",
        "import numpy.random as npr",
        "import numpy as np\nnp.random.default_rng(0)",
        "import numpy as xp\nrng = xp.random",
        "import numpy\nnumpy.random.seed(1)",
    ],
)
def test_numpy_random_detector_catches(source):
    assert _numpy_random_uses(source)


def test_numpy_random_detector_ignores_plain_numpy():
    assert not _numpy_random_uses("import numpy as np\nnp.argsort([1])\nrandom = 1\nx.random")


def test_no_numpy_random_in_arms():
    for path in _files():
        hits = _numpy_random_uses(path.read_text(encoding="utf-8"))
        assert not hits, f"{path.name}: {hits}"
