import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_committee.data import (
    from_jsonl_bytes,
    sha256_hex,
    split_cases,
    to_jsonl_bytes,
    write_split,
)
from agent_committee.schemas import Case
from agent_committee.seal import (
    ARMS_PATH,
    FROZEN_PATHS,
    SealError,
    check_unseal_conditions,
    load_test,
)

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
    for name in ("llm.py", "data.py", "schemas.py", "prereg.py", "seal.py"):
        (tmp_path / "src" / "agent_committee" / name).write_text("x = 0\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
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


@pytest.mark.parametrize("commit_the_change", [False, True])
def test_replacing_both_sealed_file_and_hash_file_fails(repo, commit_the_change):
    """Attacker swaps TEST and its hash together; the hash frozen at the tag still wins."""
    git(repo, "tag", "arms-frozen")
    fake = to_jsonl_bytes([Case(id="x", text="swapped", label="a")])
    (repo / "sealed" / "test.jsonl").write_bytes(fake)
    (repo / "data" / "test.sha256").write_text(sha256_hex(fake) + "\n")
    if commit_the_change:
        commit_all(repo, "swap test set and hash")
    with pytest.raises(SealError):
        load_test(repo)


def test_load_test_compares_against_the_frozen_hash_even_if_diff_check_is_bypassed(
    repo, monkeypatch
):
    git(repo, "tag", "arms-frozen")
    monkeypatch.setattr("agent_committee.seal.check_unseal_conditions", lambda _repo: None)
    fake = to_jsonl_bytes([Case(id="x", text="swapped", label="a")])
    (repo / "sealed" / "test.jsonl").write_bytes(fake)
    (repo / "data" / "test.sha256").write_text(sha256_hex(fake) + "\n")  # self-consistent forgery
    with pytest.raises(SealError, match="differs from tag"):
        load_test(repo)


@pytest.mark.parametrize("rel", [p for p in FROZEN_PATHS if p != ARMS_PATH])
def test_refuses_if_any_frozen_file_changed_after_freeze(repo, rel):
    git(repo, "tag", "arms-frozen")
    p = repo / rel
    assert p.is_file(), f"fixture must contain frozen file {rel}"
    p.write_bytes(p.read_bytes() + b"\n# edit after freeze\n")
    with pytest.raises(SealError, match="changed since"):
        check_unseal_conditions(repo)


def test_frozen_paths_include_the_guard_itself_and_the_build_config():
    for must in (
        "src/agent_committee/seal.py",
        "src/agent_committee/prereg.py",
        "pyproject.toml",
        "uv.lock",
        "data/split_manifest.json",
    ):
        assert must in FROZEN_PATHS


def test_load_test_works_with_absolute_repo_data_and_sealed_paths(repo):
    git(repo, "tag", "arms-frozen")
    expected = from_jsonl_bytes((repo / "sealed" / "test.jsonl").read_bytes())
    got = load_test(
        repo.resolve(),
        data_dir=(repo / "data").resolve(),
        sealed_dir=(repo / "sealed").resolve(),
    )
    assert got == expected


def test_load_test_rejects_data_dir_outside_the_repo(repo, tmp_path_factory):
    git(repo, "tag", "arms-frozen")
    with pytest.raises(SealError, match="outside the repository"):
        load_test(repo, data_dir=tmp_path_factory.mktemp("elsewhere"))


def test_refuses_if_dev_data_changed_after_freeze(repo):
    git(repo, "tag", "arms-frozen")
    p = repo / "data" / "dev.jsonl"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(SealError, match="changed since"):
        check_unseal_conditions(repo)


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
