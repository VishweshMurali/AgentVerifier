import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_committee.data import from_jsonl_bytes, split_cases, write_split
from agent_committee.schemas import Case
from agent_committee.seal import SealError, check_unseal_conditions, load_test

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "agent_committee"


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def commit_all(repo, msg="c"):
    git(repo, "add", "-A")
    git(repo, "commit", "-m", msg)


@pytest.fixture
def repo(tmp_path):
    """A git repo: prereg.yaml, an arms package, a committed hash, a gitignored sealed file."""
    git(tmp_path, "init", "-b", "main")
    shutil.copy(ROOT / "prereg.yaml", tmp_path / "prereg.yaml")
    arms = tmp_path / "src" / "agent_committee" / "arms"
    arms.mkdir(parents=True)
    (arms / "__init__.py").write_text("")
    (tmp_path / ".gitignore").write_text("sealed/\n")
    cases = [Case(id=f"{x}{i}", text=f"t {x} {i}", label=x) for x in "ab" for i in range(10)]
    dev, test = split_cases(cases, 0.25, seed=1)
    write_split(
        dev,
        test,
        seed=1,
        test_fraction=0.25,
        data_dir=tmp_path / "data",
        sealed_dir=tmp_path / "sealed",
    )
    commit_all(tmp_path, "init")
    return tmp_path


def test_refuses_without_freeze_tag(repo):
    with pytest.raises(SealError, match="does not exist"):
        load_test(repo)


def test_unseals_when_frozen_and_clean(repo):
    git(repo, "tag", "arms-frozen")
    expected = from_jsonl_bytes((repo / "sealed" / "test.jsonl").read_bytes())
    assert load_test(repo) == expected and len(expected) == 5


def test_refuses_tampered_sealed_file(repo):
    git(repo, "tag", "arms-frozen")
    p = repo / "sealed" / "test.jsonl"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(SealError, match="SHA-256"):
        load_test(repo)


def test_refuses_if_prereg_changed_after_freeze(repo):
    git(repo, "tag", "arms-frozen")
    (repo / "prereg.yaml").write_text((repo / "prereg.yaml").read_text() + "# edit\n")
    with pytest.raises(SealError, match="changed since"):
        check_unseal_conditions(repo)
    commit_all(repo, "edit prereg after freeze")  # committing it does not help
    with pytest.raises(SealError, match="changed since"):
        check_unseal_conditions(repo)


def test_refuses_if_arms_code_changed_after_freeze(repo):
    git(repo, "tag", "arms-frozen")
    (repo / "src" / "agent_committee" / "arms" / "__init__.py").write_text("x = 1\n")
    with pytest.raises(SealError, match="changed since"):
        check_unseal_conditions(repo)


def test_refuses_untracked_new_arm_file_after_freeze(repo):
    git(repo, "tag", "arms-frozen")
    (repo / "src" / "agent_committee" / "arms" / "sneaky.py").write_text("x = 1\n")
    with pytest.raises(SealError, match="untracked"):
        check_unseal_conditions(repo)


def test_refuses_if_tag_is_not_an_ancestor_of_head(repo):
    git(repo, "checkout", "-b", "side")
    (repo / "note.txt").write_text("x")
    commit_all(repo, "side commit")
    git(repo, "tag", "arms-frozen")
    git(repo, "checkout", "main")
    with pytest.raises(SealError, match="ancestor"):
        check_unseal_conditions(repo)


def test_refuses_if_prereg_not_committed(repo):
    git(repo, "tag", "arms-frozen")
    git(repo, "rm", "--cached", "prereg.yaml")
    with pytest.raises(SealError, match="not committed"):
        check_unseal_conditions(repo)


def _module_imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names.add(mod)
            names |= {f"{mod}.{a.name}" for a in node.names}
    return names


def test_only_the_runner_imports_seal():
    allowed = {"seal.py", "runner.py"}
    for path in SRC.rglob("*.py"):
        if path.name in allowed:
            continue
        hits = {n for n in _module_imports(path) if n.endswith("seal") or ".seal." in n + "."}
        assert not hits, f"{path.relative_to(ROOT)} must not import seal: {hits}"


def test_only_data_and_seal_know_the_sealed_test_filename():
    for path in SRC.rglob("*.py"):
        if path.name in {"data.py", "seal.py"}:
            continue
        assert "SEALED_TEST_FILE" not in path.read_text(encoding="utf-8"), path.name
        assert "sealed/" not in path.read_text(encoding="utf-8"), path.name
