"""Load and validate prereg.yaml."""

from pathlib import Path

import yaml

from agent_committee.schemas import Prereg


def load_prereg(path: str | Path = "prereg.yaml", task: str | None = None) -> Prereg:
    """Parse prereg.yaml (safe_load only) into a validated Prereg. With `task`, refuse a prereg
    written for another scenario."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    prereg = Prereg.model_validate(data)
    if task is not None and prereg.task != task:
        raise ValueError(f"{path} is the prereg of scenario {prereg.task!r}, not {task!r}")
    return prereg
