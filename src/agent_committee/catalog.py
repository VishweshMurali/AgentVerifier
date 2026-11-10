"""The dataset menu: standard datasets a user can pick, plus their own file.

Choosing a preset writes its `scenarios/<name>/scenario.yaml` (that file, not this module, is
frozen with the scenario), so presets can be added here later without touching any tagged
scenario. Each preset was checked once against the real data (2026-11-10): columns, labels, and
at least 30 cases per label. Sizes shown to users come from a check or a built manifest, never
from this file.

Banking77 is built in. MASSIVE was left out: 4 of its 60 intents have fewer than 30 cases.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_committee.data import DEV_FILE, MANIFEST_FILE
from agent_committee.scenarios import (
    BANKING77,
    HuggingFaceSource,
    Scenario,
    ScenarioFile,
    add_scenario,
    definition_path,
    loader_for,
    read_scenario_file,
)

# until each dataset has its own task description, the prompt and agent arms speak of banking,
# so a new dataset is blocked from the runner and the seal (arms_ready=False)
NEW_DATASET_ARMS = (
    "DEV preview with non-LLM arms only (majority, linear, Naive Bayes); no rehearsal or verdict "
    "until a per-dataset task phrase exists"
)
BANKING77_ARMS = "all arms; DEV rehearsals only, its sealed test was used in v1"


@dataclass(frozen=True)
class Preset:
    name: str
    title: str
    description: str
    spec: ScenarioFile | None  # None: built in (Banking77)


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        Preset(
            "banking77",
            "Banking77: bank support routing",
            "Customer messages to a bank, routed to one of 77 intents. The v1 scenario.",
            None,
        ),
        Preset(
            "clinc150",
            "CLINC150: assistant intents with out-of-scope",
            "Requests to a virtual assistant across 150 intents, plus 'oos' for anything out of "
            "scope (scored as an ordinary label).",
            ScenarioFile(
                name="clinc150",
                description="CLINC150 (clinc/clinc_oos, plus), all splits pooled",
                source=HuggingFaceSource(
                    kind="huggingface",
                    dataset="clinc/clinc_oos",
                    config="plus",
                    text="text",
                    label="intent",
                    splits=["train", "validation", "test"],
                    revision="155b9c710419136e17307b80d0a13e68cd46b4ec",
                ),
            ),
        ),
        Preset(
            "emotion",
            "Emotion: tone of short messages",
            "Short messages labelled with one of 6 emotions, as in support triage by tone.",
            ScenarioFile(
                name="emotion",
                description="Emotion (dair-ai/emotion, split), all splits pooled",
                source=HuggingFaceSource(
                    kind="huggingface",
                    dataset="dair-ai/emotion",
                    config="split",
                    text="text",
                    label="label",
                    splits=["train", "validation", "test"],
                    revision="cab853a1dbdf4c42c2b3ef2173804746df8825fe",
                ),
            ),
        ),
    )
}
BRING_YOUR_OWN = {
    "option": "own",
    "title": "Your own dataset",
    "description": "A .jsonl or .csv under sources/ with id, text, label (and optional group) "
    "columns; add it with add_dataset.",
}


def _status(preset: Preset, repo: Path) -> dict[str, Any]:
    if preset.spec is None:
        s: Scenario = BANKING77
        chosen = True
    else:
        chosen = (repo / definition_path(preset.name)).is_file()
        s = Scenario.standard(preset.name, lambda: [], definition=definition_path(preset.name))
    built = (repo / s.data_dir / DEV_FILE).is_file()
    if preset.spec is None:
        status = "built in"
    elif chosen and read_scenario_file(preset.name, repo) != preset.spec:
        # a scenario of this name exists but is not this preset (e.g. someone's own file)
        status = f"conflict: {definition_path(preset.name)} is not this preset"
    else:
        status = "split built" if built else "chosen" if chosen else "available"
    out: dict[str, Any] = {
        "option": preset.name,
        "title": preset.title,
        "description": preset.description,
        "status": status,
        "arms": BANKING77_ARMS if s.arms_ready else NEW_DATASET_ARMS,
    }
    if built:  # sizes only from a built manifest
        manifest = json.loads((repo / s.data_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        out["cases"] = manifest["n_dev"] + manifest["n_test"]
        out["labels"] = manifest["n_labels"]
    return out


def menu(repo: str | Path = ".") -> list[dict[str, Any]]:
    """Every standard option with its status, then 'your own dataset'."""
    repo = Path(repo)
    return [*(_status(p, repo) for p in PRESETS.values()), dict(BRING_YOUR_OWN)]


def choose(option: str, repo: str | Path = ".") -> dict[str, Any]:
    """Pick a standard dataset: writes its scenario.yaml once. Downloads nothing and builds no
    split. Choosing it again, or choosing Banking77, only reports its status."""
    repo = Path(repo)
    if option not in PRESETS:
        known = sorted(PRESETS)
        raise ValueError(f"unknown option {option!r}; choose one of {known} or add your own")
    preset = PRESETS[option]
    if preset.spec is not None and not (repo / definition_path(option)).is_file():
        add_scenario(preset.spec, repo)
    return _status(preset, repo)


def main(argv: list[str] | None = None) -> int:
    """`python -m agent_committee.catalog` prints the menu; `--choose NAME` picks an option."""
    import argparse

    ap = argparse.ArgumentParser(description="The dataset menu.")
    ap.add_argument("--choose", default=None)
    ap.add_argument("--repo", default=".")
    args = ap.parse_args(argv)
    if args.choose:
        print(json.dumps(choose(args.choose, args.repo), indent=2))
        return 0
    for m in menu(args.repo):
        print(f"{m['option']:10} {m.get('status', ''):12} {m['title']}")
    return 0


def preload(repo: str | Path = ".") -> dict[str, str]:
    """Download every preset into the Hugging Face cache, so choosing one later is instant.
    Writes no scenario file and builds no split; reports per preset, never raises."""
    out = {}
    for name, preset in PRESETS.items():
        if preset.spec is None:
            continue
        try:
            n = len(loader_for(preset.spec, Path(repo))())
            out[name] = f"ready: {n} cases"
        except Exception as exc:  # offline, rate-limited: the server still starts
            out[name] = f"failed: {exc!r}"[:300]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
