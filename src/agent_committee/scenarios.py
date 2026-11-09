"""Scenarios: a labelled dataset plugged into the verifier with its own prereg, split and seal.

A scenario names where everything lives (prereg, DEV data, DEV report, results) and how its
cases are loaded. Where its sealed TEST lives is known only to data.py and seal.py
(`data.sealed_dir`). Banking77 is scenario 1 and keeps the exact v1 paths and
the `arms-frozen` tag, so its v1 split, hashes and verdict reproduce unchanged. Any other scenario
uses its own folders (`Scenario.standard`).

This module is frozen with the arms (seal.FROZEN_PATHS): it decides which files the unseal guard
checks, so it must not change after a scenario's freeze tag. It never reads TEST.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from agent_committee.schemas import Case

Loader = Callable[[], list[Case]]
NAME = re.compile(r"^[a-z0-9_]+$")  # the same rule as Prereg.task
# results/rehearsal/ is Banking77's rehearsal folder, which the server reads
RESERVED_NAMES = {"rehearsal"}


@dataclass(frozen=True)
class Scenario:
    name: str
    freeze_tag: str
    prereg_path: str
    data_dir: str
    dev_report: str
    results_dir: str
    loader: Loader
    # the arms have this scenario's own config (null threshold, shortlist, model digest, DEV
    # report). Until they do, the runner and the unseal guard refuse it (arms config is P3).
    arms_ready: bool = False

    @classmethod
    def standard(cls, name: str, loader: Loader) -> "Scenario":
        """A new scenario in its own folders, frozen by the tag `arms-frozen-<name>`."""
        if not NAME.match(name):
            raise ValueError(f"scenario name must be lowercase letters, digits and _: {name!r}")
        if name in RESERVED_NAMES:
            raise ValueError(f"scenario name {name!r} is reserved")
        return cls(
            name=name,
            freeze_tag=f"arms-frozen-{name}",
            prereg_path=f"scenarios/{name}/prereg.yaml",
            data_dir=f"data/{name}",
            dev_report=f"scenarios/{name}/dev_report.json",
            results_dir=f"results/{name}",
            loader=loader,
        )


def _banking77() -> list[Case]:
    from agent_committee.data import load_banking77

    return load_banking77()


BANKING77 = Scenario(
    name="banking77",
    freeze_tag="arms-frozen",
    prereg_path="prereg.yaml",
    data_dir="data",
    dev_report="src/agent_committee/arms/dev_report.json",
    results_dir="results",
    loader=_banking77,
    arms_ready=True,
)

SCENARIOS: dict[str, Scenario] = {BANKING77.name: BANKING77}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; registered: {sorted(SCENARIOS)}")
    return SCENARIOS[name]
