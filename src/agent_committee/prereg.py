"""Load and validate prereg.yaml."""

from pathlib import Path

import yaml

from agent_committee.schemas import Prereg


def load_prereg(path: str | Path = "prereg.yaml") -> Prereg:
    """Parse prereg.yaml (safe_load only) into a validated Prereg."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Prereg.model_validate(data)
