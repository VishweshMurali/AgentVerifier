"""The ONLY module that reads the sealed TEST set. Only the Phase D runner may import it.

Unsealing requires ALL of:
 1. git tag `arms-frozen` exists
 2. the tag is an ancestor of HEAD
 3. prereg.yaml is tracked, committed, and unchanged since the tag
 4. nothing under src/agent_committee/arms/ changed since the tag (tracked or untracked)
 5. sealed/test.jsonl matches the SHA-256 committed in data/test.sha256

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

    # Tracked changes (staged or not) vs the tag, for the two frozen paths.
    if _git(repo, "diff", "--quiet", FREEZE_TAG, "--", PREREG_PATH, ARMS_PATH).returncode != 0:
        raise SealError(f"{PREREG_PATH} or {ARMS_PATH} changed since tag '{FREEZE_TAG}'")

    # Untracked files under arms/ would not show up in `git diff`.
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "--", ARMS_PATH)
    if untracked.stdout.strip():
        raise SealError(
            f"untracked files under {ARMS_PATH} since freeze: {untracked.stdout.strip()}"
        )


def load_test(
    repo: str | Path = ".",
    data_dir: str | Path = "data",
    sealed_dir: str | Path = "sealed",
) -> list[Case]:
    """Verify unseal conditions + hash, then return the TEST cases."""
    repo = Path(repo)
    check_unseal_conditions(repo)

    raw = (repo / sealed_dir / SEALED_TEST_FILE).read_bytes()
    expected = (repo / data_dir / TEST_HASH_FILE).read_text(encoding="utf-8").strip()
    if sha256_hex(raw) != expected:
        raise SealError("sealed test file does not match the committed SHA-256")
    return from_jsonl_bytes(raw)
