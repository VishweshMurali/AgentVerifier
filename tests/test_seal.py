import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_committee.arms.config import MODEL_DIGEST, NULL_THRESHOLD
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
    DEV_REPORT,
    FROZEN_PATHS,
    REQUIRED_ARM_FILES,
    SealError,
    check_pretag,
    check_unseal_conditions,
    load_test,
    rebuild_fingerprint,
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
def repo(tmp_path, monkeypatch):
    """A git repo: prereg.yaml, an arms package, a committed hash, a gitignored sealed file."""
    # the fixture's DEV data is too small to split, so the rebuild itself is tested on the real pool
    monkeypatch.setattr("agent_committee.seal.rebuild_fingerprint", lambda _repo: "fp")
    git(tmp_path, "init", "-b", "main")
    shutil.copy(ROOT / "prereg.yaml", tmp_path / "prereg.yaml")
    arms = tmp_path / "src" / "agent_committee" / "arms"
    arms.mkdir(parents=True)
    (arms / "__init__.py").write_text("")
    for name in ("base.py", "linear.py", "prompt.py", "agent.py"):
        (arms / name).write_text("x = 0\n")
    (arms / "dev_report.json").write_text(
        json.dumps(
            {
                "results": {"linear": {"accuracy": 0.9}},
                "model_digest": MODEL_DIGEST,
                "layer": {"fingerprint": "fp"},
            }
        )
    )
    for name in ("llm.py", "data.py", "schemas.py", "scorers.py", "prereg.py", "seal.py"):
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


@pytest.mark.parametrize("rel", [*REQUIRED_ARM_FILES, DEV_REPORT])
def test_refuses_if_a_required_file_is_missing_at_the_tag(repo, rel):
    git(repo, "rm", "-q", rel)
    git(repo, "commit", "-m", "drop")
    git(repo, "tag", "arms-frozen")
    with pytest.raises(SealError, match="missing at tag"):
        check_unseal_conditions(repo)


def test_refuses_if_arms_directory_is_empty_at_the_tag(repo):
    git(repo, "rm", "-rq", ARMS_PATH)
    git(repo, "commit", "-m", "drop arms")
    git(repo, "tag", "arms-frozen")
    with pytest.raises(SealError, match="missing at tag"):
        check_unseal_conditions(repo)


def test_refuses_dirty_or_untracked_file_in_src_or_scripts(repo):
    git(repo, "tag", "arms-frozen")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "extra.py").write_text("x = 1\n")
    with pytest.raises(SealError, match="uncommitted"):
        check_unseal_conditions(repo)


def test_branch_named_like_the_tag_cannot_stand_in_for_it(repo):
    git(repo, "branch", "arms-frozen")
    with pytest.raises(SealError, match="does not exist"):
        check_unseal_conditions(repo)


def test_pretag_ready_then_flags_problems(repo):
    assert check_pretag(repo) == []
    (repo / DEV_REPORT).write_text(json.dumps({"results": {"a": 1}, "model_digest": "other"}))
    assert any("model_digest" in p for p in check_pretag(repo))
    (repo / DEV_REPORT).write_text("not json")
    assert any("not a report" in p for p in check_pretag(repo))
    commit_all(repo, "bad report")
    (repo / "src" / "agent_committee" / "arms" / "new.py").write_text("y = 1\n")
    assert any("uncommitted" in p for p in check_pretag(repo))


def test_pretag_flags_missing_files_and_existing_tag(repo):
    git(repo, "rm", "-q", REQUIRED_ARM_FILES[0])
    git(repo, "commit", "-m", "drop")
    assert any("not committed at HEAD" in p for p in check_pretag(repo))
    git(repo, "tag", "arms-frozen")
    assert any("already exists" in p for p in check_pretag(repo))


def test_pretag_never_reads_the_sealed_file(repo):
    (repo / "sealed" / "test.jsonl").unlink()
    assert check_pretag(repo) == []


def test_unseal_refuses_when_the_rebuilt_layer_differs(repo, monkeypatch):
    git(repo, "tag", "arms-frozen")
    monkeypatch.setattr("agent_committee.seal.rebuild_fingerprint", lambda _repo: "other")
    with pytest.raises(SealError, match="does not match"):
        check_unseal_conditions(repo)


def test_pretag_flags_a_rebuilt_layer_that_differs(repo, monkeypatch):
    monkeypatch.setattr("agent_committee.seal.rebuild_fingerprint", lambda _repo: "other")
    assert any("does not match" in p for p in check_pretag(repo))


def test_pretag_flags_a_report_with_no_fingerprint(repo):
    report = {"results": {"linear": {}}, "model_digest": MODEL_DIGEST}
    (repo / DEV_REPORT).write_text(json.dumps(report))
    assert any("no layer fingerprint" in p for p in check_pretag(repo))


def test_rebuild_fingerprint_is_stable_on_the_real_pool():
    assert rebuild_fingerprint(ROOT) == rebuild_fingerprint(ROOT)


def test_pretag_and_unseal_refuse_a_prereg_missing_a_section(repo):
    text = (ROOT / "prereg.yaml").read_text(encoding="utf-8")
    (repo / "prereg.yaml").write_text(text.split("\nlinear:")[0] + "\n", encoding="utf-8")
    commit_all(repo, "prereg without the new sections")
    problems = check_pretag(repo)
    for name in ("linear", "scorer", "quality_floor"):
        assert any(f"does not set {name}" in p for p in problems)
    git(repo, "tag", "arms-frozen")
    with pytest.raises(SealError, match="does not set"):
        check_unseal_conditions(repo)


def test_pretag_flags_prereg_linear_settings_that_differ_from_config(repo):
    prereg = repo / "prereg.yaml"
    prereg.write_text(prereg.read_text().replace(str(NULL_THRESHOLD), "0.1"))
    assert any("do not match arms/config" in p for p in check_pretag(repo))
