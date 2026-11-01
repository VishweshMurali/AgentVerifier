"""The ONLY module that reads the sealed TEST set. Only the Phase D runner may import it.

Unsealing requires ALL of:
 1. git tag `arms-frozen` exists
 2. the tag is an ancestor of HEAD
 3. prereg.yaml is tracked and committed
 4. nothing in FROZEN_PATHS (prereg, arms/, core modules incl. this one, pyproject/uv.lock,
    DEV data, hashes, manifest) changed since the tag, tracked or untracked
 5. sealed/test.jsonl matches the SHA-256 recorded in data/test.sha256 AS OF THE TAG

This is tamper-EVIDENT, not tamper-proof: the repo owner can still create the tag. What it
guarantees that an honest run cannot unseal by accident and any edit after freezing is caught.
"""

import subprocess
from pathlib import Path

from agent_committee.data import (
    SEALED_TEST_FILE,
    TEST_HASH_FILE,
    from_jsonl_bytes,
    sha256_hex,
)
from agent_committee.schemas import Case

FREEZE_TAG = "arms-frozen"
PREREG_PATH = "prereg.yaml"
ARMS_PATH = "src/agent_committee/arms"
# Everything that must be byte-identical between the freeze tag and unsealing.
FROZEN_PATHS = [
    PREREG_PATH,
    ARMS_PATH,
    "src/agent_committee/llm.py",
    "src/agent_committee/data.py",
    "src/agent_committee/schemas.py",
    "src/agent_committee/prereg.py",
    "src/agent_committee/seal.py",
    "pyproject.toml",
    "uv.lock",
    "data/dev.jsonl",
    "data/dev.sha256",
    "data/test.sha256",
    "data/split_manifest.json",
]


class SealError(RuntimeError):
    """Raised when any unseal precondition fails."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)


def check_unseal_conditions(repo: str | Path = ".") -> None:
    """Raise SealError unless the arms are provably frozen and the pre-registration is committed."""
    repo = Path(repo)

    if _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{FREEZE_TAG}").returncode != 0:
        raise SealError(f"git tag '{FREEZE_TAG}' does not exist; arms are not frozen")

    if _git(repo, "merge-base", "--is-ancestor", FREEZE_TAG, "HEAD").returncode != 0:
        raise SealError(f"tag '{FREEZE_TAG}' is not an ancestor of HEAD")

    if _git(repo, "ls-files", "--error-unmatch", PREREG_PATH).returncode != 0:
        raise SealError(f"{PREREG_PATH} is not committed")

    # Tracked changes (staged or not) vs the tag, for every frozen path.
    if _git(repo, "diff", "--quiet", FREEZE_TAG, "--", *FROZEN_PATHS).returncode != 0:
        raise SealError(f"frozen paths changed since tag '{FREEZE_TAG}': {FROZEN_PATHS}")

    # Untracked files under frozen paths would not show up in `git diff`.
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "--", *FROZEN_PATHS)
    if untracked.stdout.strip():
        raise SealError(f"untracked files under frozen paths since freeze: {untracked.stdout}")


def load_test(
    repo: str | Path = ".",
    data_dir: str | Path = "data",
    sealed_dir: str | Path = "sealed",
) -> list[Case]:
    """Verify unseal conditions + hash, then return the TEST cases.

    The expected hash comes from the FROZEN revision (`git show arms-frozen:...`), not the working
    tree, so replacing both the sealed file and the working-tree hash file cannot pass.
    """
    repo = Path(repo)
    check_unseal_conditions(repo)

    # `git show tag:path` needs a repo-relative POSIX path; data_dir may be absolute.
    try:
        hash_path = (repo / data_dir / TEST_HASH_FILE).resolve().relative_to(repo.resolve())
    except ValueError as exc:
        raise SealError(f"data_dir {data_dir} is outside the repository") from exc
    hash_path = hash_path.as_posix()
    frozen = _git(repo, "show", f"{FREEZE_TAG}:{hash_path}")
    if frozen.returncode != 0:
        raise SealError(f"{hash_path} does not exist in tag '{FREEZE_TAG}'")
    expected = frozen.stdout.strip()

    if (repo / hash_path).read_text(encoding="utf-8").strip() != expected:
        raise SealError(f"{hash_path} in the working tree differs from tag '{FREEZE_TAG}'")

    raw = (repo / sealed_dir / SEALED_TEST_FILE).read_bytes()
    if sha256_hex(raw) != expected:
        raise SealError("sealed test file does not match the frozen SHA-256")
    return from_jsonl_bytes(raw)
