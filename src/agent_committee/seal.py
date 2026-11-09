"""The ONLY module that reads the sealed TEST set. Only the Phase D runner may import it.

Every check is per scenario (scenarios.py); the defaults are Banking77's v1 paths and tag.
Unsealing requires ALL of:
 0. the arms have the scenario's own config (`Scenario.arms_ready`)
 1. the scenario's freeze tag (`arms-frozen` for Banking77) exists
 2. the tag is an ancestor of HEAD
 3. prereg.yaml is tracked and committed
 4. nothing in FROZEN_PATHS (prereg, arms/, core modules incl. this one, pyproject/uv.lock,
    DEV data, hashes, manifest) changed since the tag, tracked or untracked
 5. prereg.yaml sets the linear layer, scorer, quality floor and checks; the layer matches config
 6. the linear layer rebuilt from the pool has the fingerprint in the DEV report
 7. sealed/test.jsonl matches the SHA-256 recorded in data/test.sha256 AS OF THE TAG

This is tamper-EVIDENT, not tamper-proof: the repo owner can still create the tag. What it
guarantees that an honest run cannot unseal by accident and any edit after freezing is caught.
"""

import argparse
import json
import subprocess
from pathlib import Path

from agent_committee.arms.config import MODEL_DIGEST, NULL_THRESHOLD, SHORTLIST_K
from agent_committee.arms.linear import make_layer
from agent_committee.arms.splits import make_arms_split
from agent_committee.data import (
    SEALED_TEST_FILE,
    TEST_HASH_FILE,
    from_jsonl_bytes,
    load_dev,
    sha256_hex,
)
from agent_committee.data import sealed_dir as sealed_dir_for
from agent_committee.prereg import load_prereg
from agent_committee.scenarios import BANKING77, Scenario
from agent_committee.schemas import Case

ARMS_PATH = "src/agent_committee/arms"
# Must exist at the tag on top of every frozen path, so an empty arms/ cannot pass.
REQUIRED_ARM_FILES = [f"{ARMS_PATH}/{n}" for n in ("base.py", "linear.py", "prompt.py", "agent.py")]
# Code that must be byte-identical between any scenario's freeze tag and unsealing.
FROZEN_CODE = [
    ARMS_PATH,
    "src/agent_committee/scenarios.py",
    "src/agent_committee/llm.py",
    "src/agent_committee/data.py",
    "src/agent_committee/schemas.py",
    "src/agent_committee/scorers.py",
    "src/agent_committee/checks.py",
    "src/agent_committee/prereg.py",
    "src/agent_committee/seal.py",
    # the Phase D code that turns the paired run into a verdict is frozen with everything else
    "src/agent_committee/runner.py",
    "src/agent_committee/analysis.py",
    "src/agent_committee/decision.py",
    "src/agent_committee/report.py",
    "src/agent_committee/templates",
    "pyproject.toml",
    "uv.lock",
]
DATA_FILES = ("dev.jsonl", "dev.sha256", TEST_HASH_FILE, "split_manifest.json")


def tag_ref(s: Scenario) -> str:
    return f"refs/tags/{s.freeze_tag}"


def frozen_paths(s: Scenario) -> list[str]:
    """Everything that must be byte-identical between the scenario's freeze tag and unsealing:
    its prereg, the shared code (scenarios.py included) and its DEV data, hashes and manifest."""
    data = [f"{s.data_dir}/{name}" for name in DATA_FILES]
    # Banking77's DEV report lives in arms/ (already frozen); any other scenario's is listed itself
    report = [] if s.dev_report.startswith(f"{ARMS_PATH}/") else [s.dev_report]
    return [s.prereg_path, *FROZEN_CODE, *data, *report]


def required_at_tag(s: Scenario) -> list[str]:
    return [*frozen_paths(s), *REQUIRED_ARM_FILES, s.dev_report]


# Banking77 (v1) names, kept for callers and tests
FREEZE_TAG = BANKING77.freeze_tag
PREREG_PATH = BANKING77.prereg_path
TAG_REF = tag_ref(BANKING77)
DEV_REPORT = BANKING77.dev_report
FROZEN_PATHS = frozen_paths(BANKING77)
REQUIRED_AT_TAG = required_at_tag(BANKING77)
# Source trees that must have no uncommitted change at all when unsealing.
CLEAN_TREES = ["src", "scripts"]


class SealError(RuntimeError):
    """Raised when any unseal precondition fails."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)


def rebuild_fingerprint(repo: Path, data_dir: str = BANKING77.data_dir) -> str:
    """Fit the layer again from the committed DEV pool. Reads DEV only."""
    pool = make_arms_split(load_dev(repo / data_dir)).pool
    return make_layer(pool).fingerprint()


def _layer_problems(repo: Path, s: Scenario = BANKING77) -> list[str]:
    """Is the frozen layer reproducible? Compares a fresh rebuild with the DEV report."""
    try:
        data = json.loads((repo / s.dev_report).read_text(encoding="utf-8"))
        recorded = data["layer"]["fingerprint"]
    except (OSError, ValueError, KeyError, TypeError):
        return [f"{s.dev_report} has no layer fingerprint"]
    if rebuild_fingerprint(repo, s.data_dir) != recorded:
        return [f"linear layer rebuilt from the pool does not match {s.dev_report}"]
    return []


def _prereg_problems(repo: Path, s: Scenario = BANKING77) -> list[str]:
    """Sections the pre-registration must settle before the tag."""
    if not s.arms_ready:
        return [f"scenario {s.name!r} has no arms config yet"]
    try:
        prereg = load_prereg(repo / s.prereg_path, task=s.name)
    except ValueError as exc:
        return [str(exc)]
    problems = [
        f"{s.prereg_path} does not set {name}"
        for name in ("linear", "scorer", "quality_floor", "checks")
        if getattr(prereg, name) is None
    ]
    if prereg.linear is not None and (
        prereg.linear.null_threshold != NULL_THRESHOLD or prereg.linear.shortlist_k != SHORTLIST_K
    ):
        problems.append(f"{s.prereg_path} linear settings do not match arms/config")
    return problems


def check_unseal_conditions(repo: str | Path = ".", s: Scenario = BANKING77) -> None:
    """Raise SealError unless the arms are provably frozen and the pre-registration is committed."""
    repo = Path(repo)
    FREEZE_TAG, TAG_REF, PREREG_PATH = s.freeze_tag, tag_ref(s), s.prereg_path  # noqa: N806
    FROZEN_PATHS, REQUIRED_AT_TAG = frozen_paths(s), required_at_tag(s)  # noqa: N806

    if not s.arms_ready:
        raise SealError(f"scenario {s.name!r} has no arms config yet; it cannot be unsealed")

    if _git(repo, "rev-parse", "--verify", "--quiet", TAG_REF).returncode != 0:
        raise SealError(f"git tag '{FREEZE_TAG}' does not exist; arms are not frozen")

    if _git(repo, "merge-base", "--is-ancestor", TAG_REF, "HEAD").returncode != 0:
        raise SealError(f"tag '{FREEZE_TAG}' is not an ancestor of HEAD")

    if _git(repo, "ls-files", "--error-unmatch", PREREG_PATH).returncode != 0:
        raise SealError(f"{PREREG_PATH} is not committed")

    # Tracked changes (staged or not) vs the tag, for every frozen path.
    if _git(repo, "diff", "--quiet", TAG_REF, "--", *FROZEN_PATHS).returncode != 0:
        raise SealError(f"frozen paths changed since tag '{FREEZE_TAG}': {FROZEN_PATHS}")

    # Untracked files under frozen paths would not show up in `git diff`.
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "--", *FROZEN_PATHS)
    if untracked.stdout.strip():
        raise SealError(f"untracked files under frozen paths since freeze: {untracked.stdout}")

    # Every required file must exist in the frozen revision (an empty arms/ must not unseal).
    missing = [
        p for p in REQUIRED_AT_TAG if _git(repo, "cat-file", "-e", f"{TAG_REF}:{p}").returncode
    ]
    if missing:
        raise SealError(f"files missing at tag '{FREEZE_TAG}': {missing}")

    # Code that runs the test must be exactly what is committed: no edits, no new files.
    dirty = _git(repo, "status", "--porcelain", "--", *CLEAN_TREES)
    if dirty.stdout.strip():
        raise SealError(f"uncommitted changes in {CLEAN_TREES}:\n{dirty.stdout}")

    # Last, because a rebuild takes seconds: settled prereg, then a reproducible layer.
    problems = _prereg_problems(repo, s) or _layer_problems(repo, s)
    if problems:
        raise SealError("; ".join(problems))


def load_test(
    repo: str | Path = ".",
    data_dir: str | Path | None = None,
    sealed_dir: str | Path | None = None,
    s: Scenario = BANKING77,
) -> list[Case]:
    """Verify unseal conditions + hash, then return the TEST cases.

    The expected hash comes from the FROZEN revision (`git show arms-frozen:...`), not the working
    tree, so replacing both the sealed file and the working-tree hash file cannot pass.
    """
    repo = Path(repo)
    data_dir = s.data_dir if data_dir is None else data_dir
    sealed_dir = sealed_dir_for(s) if sealed_dir is None else sealed_dir
    FREEZE_TAG, TAG_REF = s.freeze_tag, tag_ref(s)  # noqa: N806
    check_unseal_conditions(repo, s)

    # `git show tag:path` needs a repo-relative POSIX path; data_dir may be absolute.
    try:
        hash_path = (repo / data_dir / TEST_HASH_FILE).resolve().relative_to(repo.resolve())
    except ValueError as exc:
        raise SealError(f"data_dir {data_dir} is outside the repository") from exc
    hash_path = hash_path.as_posix()
    frozen = _git(repo, "show", f"{TAG_REF}:{hash_path}")
    if frozen.returncode != 0:
        raise SealError(f"{hash_path} does not exist in tag '{FREEZE_TAG}'")
    expected = frozen.stdout.strip()

    if (repo / hash_path).read_text(encoding="utf-8").strip() != expected:
        raise SealError(f"{hash_path} in the working tree differs from tag '{FREEZE_TAG}'")

    raw = (repo / sealed_dir / SEALED_TEST_FILE).read_bytes()
    if sha256_hex(raw) != expected:
        raise SealError("sealed test file does not match the frozen SHA-256")
    return from_jsonl_bytes(raw)


def check_pretag(repo: str | Path = ".", s: Scenario = BANKING77) -> list[str]:
    """Problems that would make tagging the scenario's freeze tag now a mistake. Empty list means
    ready.

    Never reads TEST: it only inspects git state, source files and DEV-side reports.
    """
    repo = Path(repo)
    FREEZE_TAG, TAG_REF, DEV_REPORT = s.freeze_tag, tag_ref(s), s.dev_report  # noqa: N806
    FROZEN_PATHS, REQUIRED_AT_TAG = frozen_paths(s), required_at_tag(s)  # noqa: N806
    problems: list[str] = []
    if _git(repo, "rev-parse", "--verify", "--quiet", TAG_REF).returncode == 0:
        problems.append(f"tag '{FREEZE_TAG}' already exists")
    dirty = _git(repo, "status", "--porcelain", "--", *FROZEN_PATHS, *CLEAN_TREES)
    if dirty.stdout.strip():
        problems.append(f"uncommitted or untracked changes:\n{dirty.stdout.rstrip()}")
    for rel in REQUIRED_AT_TAG:
        if _git(repo, "cat-file", "-e", f"HEAD:{rel}").returncode != 0:
            problems.append(f"not committed at HEAD: {rel}")
    report = repo / DEV_REPORT
    if report.is_file():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except ValueError:
            data = None
        if not isinstance(data, dict) or not data.get("results"):
            problems.append(f"{DEV_REPORT} is not a report with results")
        elif data.get("model_digest") != MODEL_DIGEST or not MODEL_DIGEST:
            problems.append(f"{DEV_REPORT} model_digest does not match arms/config.MODEL_DIGEST")
        else:
            problems.extend(_layer_problems(repo, s))
    return [*problems, *_prereg_problems(repo, s)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-tag readiness check (never reads TEST).")
    ap.add_argument("--pretag", action="store_true", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--scenario", default=BANKING77.name)
    args = ap.parse_args(argv)
    from agent_committee.scenarios import get_scenario

    s = get_scenario(args.scenario)
    problems = check_pretag(args.repo, s)
    for p in problems:
        print("NOT READY:", p)
    if not problems:
        print(f"ready: run `git tag {s.freeze_tag}`")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
