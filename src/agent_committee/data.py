"""Scenario loading and the deterministic stratified DEV/TEST split.

Banking77 ships with train/test; we POOL them and re-split so that DEV and TEST are drawn from one
distribution under our own seed (recorded in prereg.yaml). Other scenarios plug in a loader
(`load_huggingface`, `load_local`) and get the same split, hashes and seal (see scenarios.py).

Sealing rules (see CLAUDE.md):
- This module WRITES the sealed TEST file once, at split creation, and never returns TEST rows.
- `load_dev` is the only loader exposed to arm-building code.
- Reading TEST back is `seal.load_test`'s job, and only the Phase D runner may call it.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from agent_committee.schemas import Case

DEV_FILE = "dev.jsonl"
DEV_HASH_FILE = "dev.sha256"
TEST_HASH_FILE = "test.sha256"
MANIFEST_FILE = "split_manifest.json"
SEALED_TEST_FILE = "test.jsonl"
SEALED_ROOT = "sealed"


def sealed_dir(scenario) -> str:
    """Folder of a scenario's sealed TEST file (gitignored): `sealed/` for Banking77 (v1),
    `sealed/<name>/` for every other scenario."""
    return SEALED_ROOT if scenario.data_dir == "data" else f"{SEALED_ROOT}/{scenario.name}"


def to_jsonl_bytes(cases: list[Case]) -> bytes:
    """Canonical serialization: sorted keys, UTF-8, LF line endings. This is what gets hashed."""
    # exclude_none: an ungrouped case serialises exactly as before `group` existed (v1 hashes)
    rows = (c.model_dump(exclude_none=True) for c in cases)
    lines = (json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def from_jsonl_bytes(raw: bytes) -> list[Case]:
    return [Case.model_validate_json(line) for line in raw.decode("utf-8").splitlines() if line]


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def check_cases(cases: list[Case], *, grouped: bool, min_per_label: int) -> None:
    """Input checks before a split: unique ids, enough cases (and groups) per label, and groups
    used only when the prereg says so."""
    dupes = sorted(i for i, n in Counter(c.id for c in cases).items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate case ids: {dupes[:5]}")
    small = sorted(lbl for lbl, n in Counter(c.label for c in cases).items() if n < min_per_label)
    if small:
        raise ValueError(f"labels with fewer than {min_per_label} cases: {small[:5]}")
    has_group = [c.group is not None for c in cases]
    if not grouped and any(has_group):
        raise ValueError("cases carry groups: set split.grouped explicitly in the prereg")
    if grouped:
        if not all(has_group):
            raise ValueError("split.grouped is set, so every case needs a group")
        groups: dict[str, set[str]] = {}
        for c in cases:
            groups.setdefault(c.label, set()).add(c.group)
        few = sorted(lbl for lbl, g in groups.items() if len(g) < 2)
        if few:
            raise ValueError(f"labels in fewer than 2 groups cannot reach DEV and TEST: {few[:5]}")


def split_cases(
    cases: list[Case],
    test_fraction: float,
    seed: int,
    *,
    grouped: bool = False,
    min_per_label: int = 2,
) -> tuple[list[Case], list[Case]]:
    """Deterministic stratified split by label. Returns (dev, test), each sorted by case id.

    Grouped: whole groups go to one side (StratifiedGroupKFold, one fold of round(1 / fraction)
    as TEST), so the realised test fraction is approximate; the manifest records it.
    """
    check_cases(cases, grouped=grouped, min_per_label=min_per_label)
    if grouped:
        return _grouped_split(cases, test_fraction, seed)
    ordered = sorted(cases, key=lambda c: c.id)
    dev_idx, test_idx = train_test_split(
        range(len(ordered)),
        test_size=test_fraction,
        stratify=[c.label for c in ordered],
        random_state=seed,
    )
    return [ordered[i] for i in sorted(dev_idx)], [ordered[i] for i in sorted(test_idx)]


def load_banking77() -> list[Case]:
    """Pool Banking77 train+test from Hugging Face into Cases with stable ids."""
    from datasets import load_dataset  # heavy import, only needed when building the split

    ds = load_dataset("PolyAI/banking77")
    names = ds["train"].features["label"].names
    cases: list[Case] = []
    for split in ("train", "test"):
        for i, row in enumerate(ds[split]):
            cases.append(
                Case(id=f"b77-{split}-{i:05d}", text=row["text"], label=names[row["label"]])
            )
    return cases


def _grouped_split(
    cases: list[Case], test_fraction: float, seed: int
) -> tuple[list[Case], list[Case]]:
    from sklearn.model_selection import StratifiedGroupKFold

    ordered = sorted(cases, key=lambda c: c.id)
    folds = StratifiedGroupKFold(
        n_splits=max(2, round(1 / test_fraction)), shuffle=True, random_state=seed
    )
    dev_idx, test_idx = next(
        folds.split(range(len(ordered)), [c.label for c in ordered], [c.group for c in ordered])
    )
    dev = [ordered[i] for i in sorted(dev_idx)]
    test = [ordered[i] for i in sorted(test_idx)]
    missing = sorted({c.label for c in test} - {c.label for c in dev})
    if missing:
        # the label book is built from DEV, so no arm could ever answer these
        raise ValueError(f"labels in TEST but not in DEV after the grouped split: {missing[:5]}")
    return dev, test


def load_huggingface(
    dataset: str,
    *,
    text: str,
    label: str,
    splits: tuple[str, ...],
    id_prefix: str,
    config: str | None = None,
    group: str | None = None,
    revision: str | None = None,
) -> list[Case]:
    """Pool the given splits of a Hugging Face dataset into Cases with stable ids
    (`<id_prefix>-<split>-<row>`). Class-label columns become their names. `group` names an
    optional column whose value ties cases together (see SplitConfig.grouped)."""
    from datasets import load_dataset  # heavy import, only needed when building the split

    kwargs = {"revision": revision} if revision else {}  # never trust_remote_code
    ds = load_dataset(dataset, config, **kwargs) if config else load_dataset(dataset, **kwargs)
    cases: list[Case] = []
    for split in splits:
        feature = ds[split].features[label]
        names = getattr(feature, "names", None)
        for i, row in enumerate(ds[split]):
            value = names[row[label]] if names else str(row[label])
            tie = str(row[group]) if group else None
            case_id = f"{id_prefix}-{split}-{i:05d}"
            cases.append(Case(id=case_id, text=row[text], label=value, group=tie))
    return cases


def load_local(path: str | Path) -> list[Case]:
    """Cases from a local `.jsonl` or `.csv` with `id`, `text` and `label` columns, and an
    optional `group` column (empty means no group).

    `id` is required: ids from row order would make the split depend on the file's row order.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:  # Excel CSVs start with a BOM
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"{path}: use a .jsonl or .csv file")
    cases = []
    for n, row in enumerate(rows, 1):
        missing = [k for k in ("id", "text", "label") if not isinstance(row.get(k), str)]
        if missing:
            raise ValueError(f"{path} row {n}: needs string columns {missing}")
        tie = row.get("group") or None
        if tie is not None and not isinstance(tie, str):
            raise ValueError(f"{path} row {n}: group must be a string")
        cases.append(Case(id=row["id"], text=row["text"], label=row["label"], group=tie))
    return cases


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def write_split(
    dev: list[Case],
    test: list[Case],
    *,
    seed: int,
    test_fraction: float,
    data_dir: str | Path = "data",
    sealed_dir: str | Path = "sealed",
    min_test_n: int | None = None,
    excluded_labels: list[str] | None = None,
    n_excluded_cases: int = 0,
) -> dict:
    """Write DEV (committed), sealed TEST (gitignored) and both hashes + a manifest (committed).

    Refuses (FileExistsError, nothing written) if any target already exists: a split is created
    once, and silently regenerating it would invalidate every hash and prediction built on it.
    """
    data_dir, sealed_dir = Path(data_dir), Path(sealed_dir)
    targets = [
        data_dir / DEV_FILE,
        data_dir / DEV_HASH_FILE,
        data_dir / TEST_HASH_FILE,
        data_dir / MANIFEST_FILE,
        sealed_dir / SEALED_TEST_FILE,
    ]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise FileExistsError(f"split already exists, refusing to overwrite: {existing}")
    data_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    dev_bytes, test_bytes = to_jsonl_bytes(dev), to_jsonl_bytes(test)
    (data_dir / DEV_FILE).write_bytes(dev_bytes)
    (data_dir / DEV_HASH_FILE).write_text(sha256_hex(dev_bytes) + "\n", encoding="utf-8")
    (sealed_dir / SEALED_TEST_FILE).write_bytes(test_bytes)
    (data_dir / TEST_HASH_FILE).write_text(sha256_hex(test_bytes) + "\n", encoding="utf-8")

    dev_texts = {_norm(c.text) for c in dev}
    manifest = {
        "seed": seed,
        "test_fraction": test_fraction,
        "n_dev": len(dev),
        "n_test": len(test),
        "n_labels": len({c.label for c in dev + test}),
        "dev_sha256": sha256_hex(dev_bytes),
        "test_sha256": sha256_hex(test_bytes),
        # Exact duplicate inputs across DEV/TEST are a leakage risk worth knowing about.
        "n_test_texts_duplicated_in_dev": sum(_norm(c.text) in dev_texts for c in test),
    }
    if any(c.group is not None for c in dev + test):
        # grouped split: record what was realised, and that no group crosses DEV/TEST
        dev_groups, test_groups = {c.group for c in dev}, {c.group for c in test}
        manifest["grouped"] = True
        manifest["realised_test_fraction"] = round(len(test) / len(dev + test), 4)
        manifest["n_groups_dev"], manifest["n_groups_test"] = len(dev_groups), len(test_groups)
        manifest["n_groups_shared"] = len(dev_groups & test_groups)
    if excluded_labels:
        # the verdict covers only the labels kept: say which were left out, and how many cases
        manifest["excluded_labels"] = sorted(excluded_labels)
        manifest["n_excluded_cases"] = n_excluded_cases
    if min_test_n is not None:
        # the prereg's own power bound (checks.min_test_n), on the actual TEST size
        manifest["min_test_n"] = min_test_n
        manifest["meets_min_test_n"] = len(test) >= min_test_n
    (data_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_dev(data_dir: str | Path = "data") -> list[Case]:
    """The ONLY loader arm-building code may use. Verifies the DEV hash before returning."""
    data_dir = Path(data_dir)
    raw = (data_dir / DEV_FILE).read_bytes()
    expected = (data_dir / DEV_HASH_FILE).read_text(encoding="utf-8").strip()
    if sha256_hex(raw) != expected:
        raise ValueError(f"{DEV_FILE} does not match {DEV_HASH_FILE}")
    return from_jsonl_bytes(raw)


def build_split(scenario, repo: str | Path = ".") -> dict:
    """Load a scenario's cases and write its split once, per its own prereg."""
    from agent_committee.prereg import load_prereg

    repo = Path(repo)
    if (repo / scenario.data_dir / DEV_FILE).exists():
        # refuse before loading (a Hugging Face loader downloads); write_split checks every file
        raise FileExistsError(f"split already exists for {scenario.name!r}, refusing to rebuild")
    prereg = load_prereg(repo / scenario.prereg_path, task=scenario.name)
    cases, n_excluded = scenario_cases(scenario)
    dev, test = split_cases(
        cases,
        prereg.split.test_fraction,
        prereg.split.seed,
        grouped=prereg.split.grouped,
        min_per_label=prereg.split.min_cases_per_label,
    )
    return write_split(
        dev,
        test,
        seed=prereg.split.seed,
        test_fraction=prereg.split.test_fraction,
        data_dir=repo / scenario.data_dir,
        sealed_dir=repo / sealed_dir(scenario),
        min_test_n=prereg.checks.min_test_n if prereg.checks else None,
        excluded_labels=list(getattr(scenario, "exclude_labels", ())),
        n_excluded_cases=n_excluded,
    )


def scenario_cases(scenario) -> tuple[list[Case], int]:
    """A scenario's cases with its excluded labels left out, and how many cases that removed.
    Every excluded label must exist in the data (a typo would otherwise exclude nothing)."""
    cases = scenario.loader()
    excluded = set(getattr(scenario, "exclude_labels", ()))
    unknown = sorted(excluded - {c.label for c in cases})
    if unknown:
        raise ValueError(f"excluded labels not in the data: {unknown}")
    kept = [c for c in cases if c.label not in excluded]
    return kept, len(cases) - len(kept)


# the arms that need no task phrase (arms.evaluate.CLASSIC; a test keeps the two equal)
NO_LLM_ARMS = ("majority", "linear", "naive_bayes", "linear_svm", "random_forest", "boosting")


def dev_run_command(scenario) -> str:
    """The exact DEV-run command for a built split: every arm with the scenario's own task phrase
    (non-LLM arms only without one). Plain double-quoted strings work in bash and PowerShell."""
    base = (
        f"uv run python -m agent_committee.arms.evaluate --data-dir {scenario.data_dir} "
        f"--out {scenario.dev_report}"
    )
    phrase = getattr(scenario, "task_phrase", None)
    if not phrase:
        return base + " --arms " + ",".join(NO_LLM_ARMS)
    flags = {"item": "--task-item", "items": "--task-items", "label_kind": "--task-label-kind",
             "hint": "--task-hint"}  # fmt: skip
    if any('"' in v for v in phrase.values()):
        raise ValueError("task phrase strings must not contain double quotes")
    return base + "".join(f' {flags[k]} "{v}"' for k, v in phrase.items())


def check_scenario(scenario, repo: str | Path = ".") -> dict:
    """Dry run before a split: load the cases and run the input checks. Writes nothing.

    Uses the scenario's prereg split settings when it exists, else the defaults (and says so).
    Counts only: no case text leaves this function.
    """
    from agent_committee.prereg import load_prereg
    from agent_committee.schemas import SplitConfig

    repo = Path(repo)
    prereg_file = repo / scenario.prereg_path
    prereg, prereg_state = None, "missing: split defaults used"
    if prereg_file.is_file():
        try:
            prereg, prereg_state = load_prereg(prereg_file, task=scenario.name), "found"
        except ValueError as exc:  # e.g. a template with TODOs still in it
            prereg_state = f"invalid, split defaults used: {str(exc).splitlines()[0]}"
    split = prereg.split if prereg else SplitConfig(seed=0, test_fraction=0.25)
    try:
        cases, n_excluded = scenario_cases(scenario)
        load_problems: list[str] = []
    except ValueError as exc:  # e.g. a typo in exclude_labels: reported, not raised
        cases, n_excluded, load_problems = [], 0, [str(exc)]
    per_label = Counter(c.label for c in cases)
    groups = {c.group for c in cases if c.group is not None}
    try:
        check_cases(cases, grouped=split.grouped, min_per_label=split.min_cases_per_label)
        problems: list[str] = list(load_problems)
    except ValueError as exc:
        problems = [*load_problems, str(exc)]
    expected_test = round(len(cases) * split.test_fraction)
    min_test_n = prereg.checks.min_test_n if prereg and prereg.checks else None
    return {
        "scenario": scenario.name,
        "prereg": prereg_state,
        "n_cases": len(cases),
        "n_labels": len(per_label),
        "excluded_labels": sorted(getattr(scenario, "exclude_labels", ())),
        "n_excluded_cases": n_excluded,
        "min_cases_per_label": min(per_label.values()) if per_label else 0,
        "max_cases_per_label": max(per_label.values()) if per_label else 0,
        "n_groups": len(groups),
        "grouped": split.grouped,
        "required_min_per_label": split.min_cases_per_label,
        "expected_test_n": expected_test,
        "min_test_n": min_test_n,
        "meets_min_test_n": None if min_test_n is None else expected_test >= min_test_n,
        "problems": problems,
        "split_built": (repo / scenario.data_dir / DEV_FILE).exists(),
        "dev_run_command": dev_run_command(scenario)
        if (repo / scenario.data_dir / DEV_FILE).exists()
        else None,
    }


def dev_stats(scenario, repo: str | Path = ".") -> dict:
    """Counts from a scenario's committed DEV split and its manifest. Never opens TEST."""
    repo = Path(repo)
    dev = load_dev(repo / scenario.data_dir)
    per_label = Counter(c.label for c in dev)
    manifest = json.loads((repo / scenario.data_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    keep = (
        "n_dev", "n_test", "n_labels", "seed", "test_fraction", "grouped",
        "realised_test_fraction", "n_groups_dev", "n_groups_test", "n_groups_shared",
        "min_test_n", "meets_min_test_n", "n_test_texts_duplicated_in_dev",
        "excluded_labels", "n_excluded_cases",
    )  # fmt: skip
    return {
        "scenario": scenario.name,
        "n_dev": len(dev),
        "n_labels_dev": len(per_label),
        "min_cases_per_label_dev": min(per_label.values()),
        "max_cases_per_label_dev": max(per_label.values()),
        "manifest": {k: manifest[k] for k in keep if k in manifest},
    }


def main(argv: list[str] | None = None) -> None:
    """Build a scenario's split (once): `python -m agent_committee.data [--scenario NAME]`."""
    import argparse

    from agent_committee.scenarios import get_scenario

    ap = argparse.ArgumentParser(description="Build a scenario's DEV/TEST split (once).")
    ap.add_argument("--scenario", default="banking77")
    args = ap.parse_args(argv)
    manifest = build_split(get_scenario(args.scenario, "."))
    print(json.dumps(manifest, indent=2))
    if manifest.get("meets_min_test_n") is False:
        print(
            f"WARNING: {manifest['n_test']} test cases is below the prereg's min_test_n "
            f"{manifest['min_test_n']}; the sealed run would end NEED_MORE_DATA"
        )


if __name__ == "__main__":
    main()
