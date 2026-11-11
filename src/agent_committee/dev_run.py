"""The DEV run of a scenario by name (C9-Q1).

    python -m agent_committee.dev_run --scenario NAME [--arms NAMES] [--repo DIR]

Reads the scenario's DEV folder, report path and task phrase from its own files and calls
`agent_committee.arms.evaluate.main` in-process with an argument list, so no task text ever passes
through a shell. It lives outside `arms/`: arm code may not import the scenario registry
(tests/test_arms_isolation.py). Folder and report always come from `--repo` (Banking77 too, with
its built-in v1 phrase). Without a task phrase, pass only the non-LLM arms
(`check_dataset` prints the exact command).
"""

import argparse
import sys
from pathlib import Path

from agent_committee.arms.config import BANKING77_TASK
from agent_committee.scenarios import BANKING77, get_scenario

PHRASE_FLAGS = {
    "item": "--task-item",
    "items": "--task-items",
    "label_kind": "--task-label-kind",
    "hint": "--task-hint",
}


def evaluate_args(name: str, repo: str | Path = ".", arms: str | None = None) -> list[str]:
    """The exact argument list for arms.evaluate. Each phrase value is one `--flag=value` token,
    so a value starting with '-' is never read as a flag."""
    repo = Path(repo)
    s = get_scenario(name, repo)
    args = [] if arms is None else [f"--arms={arms}"]
    # always the requested repository's paths, Banking77 included (R10-A1)
    args += [f"--data-dir={repo / s.data_dir}", f"--out={repo / s.dev_report}"]
    phrase = s.task_phrase
    if phrase is None and s.name == BANKING77.name:
        phrase = BANKING77_TASK.as_dict()  # built in: the v1 phrase, byte for byte
    for key, value in (phrase or {}).items():
        args.append(f"{PHRASE_FLAGS[key]}={value}")
    return args


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DEV run of a scenario by name.")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--repo", default=".")
    args = ap.parse_args(argv)
    try:
        forwarded = evaluate_args(args.scenario, args.repo, args.arms)
    except ValueError as exc:  # unknown scenario, bad scenario.yaml
        print(str(exc), file=sys.stderr)
        return 1
    from agent_committee.arms import evaluate  # heavy (sklearn); only when running

    return evaluate.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
