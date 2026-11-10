"""Scenarios: a labelled dataset plugged into the verifier with its own prereg, split and seal.

A scenario names where everything lives (prereg, DEV data, DEV report, results) and how its
cases are loaded. Where its sealed TEST lives is known only to data.py and seal.py
(`data.sealed_dir`). Banking77 is scenario 1 and keeps the exact v1 paths and
the `arms-frozen` tag, so its v1 split, hashes and verdict reproduce unchanged. Any other scenario
uses its own folders (`Scenario.standard`).

A new dataset can also be declared without code: `scenarios/<name>/scenario.yaml` names its source
(a Hugging Face dataset and its columns, or a local file under `sources/`, which is gitignored
because it holds TEST cases too). The file is frozen with its scenario. Nothing in it names Python
code, and Hugging Face remote code is never run.

This module is frozen with the arms (seal.FROZEN_PATHS): it decides which files the unseal guard
checks, so it must not change after a scenario's freeze tag. It never reads TEST.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

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
    # scenarios/<name>/scenario.yaml for a scenario declared by file (frozen with it); None in code
    definition: str | None = None

    @classmethod
    def standard(cls, name: str, loader: Loader, definition: str | None = None) -> "Scenario":
        """A new scenario in its own folders, frozen by the tag `arms-frozen-<name>`."""
        check_name(name)
        return cls(
            definition=definition,
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
SOURCES_DIR = "sources"


def check_name(name: str) -> None:
    if not NAME.match(name):
        raise ValueError(f"scenario name must be lowercase letters, digits and _: {name!r}")
    if name in RESERVED_NAMES:
        raise ValueError(f"scenario name {name!r} is reserved")


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HuggingFaceSource(_Spec):
    kind: Literal["huggingface"]
    dataset: str = Field(min_length=1)
    config: str | None = None
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    splits: list[str] = Field(min_length=1)
    group: str | None = None
    # the dataset commit to load (frozen with the scenario), so the data can't change under it
    revision: str | None = None


class LocalSource(_Spec):
    kind: Literal["local"]
    # relative to the repository, inside sources/ (gitignored: the file holds TEST cases too)
    path: str = Field(min_length=1)


class ScenarioFile(_Spec):
    """Contents of scenarios/<name>/scenario.yaml."""

    name: str
    description: str = ""
    source: Annotated[HuggingFaceSource | LocalSource, Field(discriminator="kind")]


def definition_path(name: str) -> str:
    return f"scenarios/{name}/scenario.yaml"


def local_source(repo: str | Path, path: str) -> Path:
    """The resolved file of a local source; refuses anything outside the repo's sources/ folder
    (also through symlinks), so no other file on the host can become DEV data."""
    root = (Path(repo) / SOURCES_DIR).resolve()
    resolved = (Path(repo) / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"local source {path!r} must be inside {SOURCES_DIR}/")
    if resolved.suffix not in (".jsonl", ".csv"):
        raise ValueError(f"local source {path!r} must be a .jsonl or .csv file")
    return resolved


def loader_for(spec: ScenarioFile, repo: str | Path = ".") -> Loader:
    """The loader a scenario file declares (downloads only when called)."""
    return _loader(spec, Path(repo))


def _loader(spec: ScenarioFile, repo: Path) -> Loader:
    from agent_committee import data

    src = spec.source
    if isinstance(src, LocalSource):
        path = local_source(repo, src.path)
        return lambda: data.load_local(path)
    return lambda: data.load_huggingface(
        src.dataset,
        text=src.text,
        label=src.label,
        splits=tuple(src.splits),
        id_prefix=spec.name,
        config=src.config,
        group=src.group,
        revision=src.revision,
    )


def read_scenario_file(name: str, repo: str | Path = ".") -> ScenarioFile:
    path = Path(repo) / definition_path(name)
    spec = ScenarioFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if spec.name != name:
        raise ValueError(f"{definition_path(name)} declares name {spec.name!r}, not {name!r}")
    return spec


def file_scenarios(repo: str | Path = ".") -> list[str]:
    """Names of the scenarios declared by file, sorted."""
    root = Path(repo) / "scenarios"
    if not root.is_dir():
        return []
    return sorted(
        p.parent.name
        for p in root.glob("*/scenario.yaml")
        if NAME.match(p.parent.name) and p.parent.name not in SCENARIOS
    )


def get_scenario(name: str, repo: str | Path = ".") -> Scenario:
    """A scenario by name: the code registry first (Banking77), then its scenario.yaml file."""
    if name in SCENARIOS:
        return SCENARIOS[name]
    check_name(name)
    if not (Path(repo) / definition_path(name)).is_file():
        known = sorted({*SCENARIOS, *file_scenarios(repo)})
        raise ValueError(f"unknown scenario {name!r}; registered: {known}")
    spec = read_scenario_file(name, repo)
    return Scenario.standard(name, _loader(spec, Path(repo)), definition=definition_path(name))


def add_scenario(spec: ScenarioFile, repo: str | Path = ".") -> Path:
    """Write scenarios/<name>/scenario.yaml once. Downloads nothing and builds no split."""
    check_name(spec.name)
    if spec.name in SCENARIOS:
        raise ValueError(f"{spec.name!r} is a built-in scenario")
    if isinstance(spec.source, LocalSource):
        local_source(repo, spec.source.path)
    path = Path(repo) / definition_path(spec.name)
    if path.exists():
        raise FileExistsError(f"{definition_path(spec.name)} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = spec.model_dump(exclude_none=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path
