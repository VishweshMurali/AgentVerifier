"""Scenarios (v2 P2): Banking77 reproduces v1 unchanged, and a new dataset plugs in end to end."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from agent_committee import runner, scenarios
from agent_committee.data import (
    build_split,
    load_dev,
    load_local,
    sealed_dir,
    split_cases,
)
from agent_committee.prereg import load_prereg
from agent_committee.scenarios import BANKING77, Scenario, get_scenario
from agent_committee.schemas import Case
from agent_committee.scorers import exact_label, get_scorer, resolve_scorer
from agent_committee.seal import FROZEN_PATHS, SealError, check_unseal_conditions, frozen_paths

ROOT = Path(__file__).resolve().parents[1]

# ---------- Banking77 is scenario 1 and reproduces v1 byte for byte ----------
V1_DEV_SHA = "58b6be3f948a2ce2a3c737385e8e6d99f1c56b728d07ae47ba8e6ee1a579d0d4"
V1_TEST_SHA = "d10c80b02189dd2f52c5255a6331a05ce5d64804604011741652937cedaa9514"
V1_PREREG_SHA = "1b83181605925cdb899559b67b728e45d9c4eb6cc7da7a48b31ebd3b97596501"
V1_LAYER_FINGERPRINT = "f5292dc4d5ae11888bc8e6ddc2f5645519a9bf833362fbe1846e672bc849563d"
V1_FROZEN_DATA = [
    "data/dev.jsonl",
    "data/dev.sha256",
    "data/test.sha256",
    "data/split_manifest.json",
]


def test_banking77_keeps_the_v1_paths_and_tag():
    assert get_scenario("banking77") is BANKING77
    assert (BANKING77.freeze_tag, BANKING77.prereg_path, BANKING77.data_dir) == (
        "arms-frozen",
        "prereg.yaml",
        "data",
    )
    assert BANKING77.dev_report == "src/agent_committee/arms/dev_report.json"
    assert BANKING77.results_dir == "results" and sealed_dir(BANKING77) == "sealed"


def test_banking77_frozen_paths_are_exactly_v1_plus_the_scenario_registry():
    assert FROZEN_PATHS == [
        "prereg.yaml",
        "src/agent_committee/arms",
        "src/agent_committee/scenarios.py",
        "src/agent_committee/llm.py",
        "src/agent_committee/data.py",
        "src/agent_committee/schemas.py",
        "src/agent_committee/scorers.py",
        "src/agent_committee/checks.py",
        "src/agent_committee/prereg.py",
        "src/agent_committee/seal.py",
        "src/agent_committee/runner.py",
        "src/agent_committee/analysis.py",
        "src/agent_committee/decision.py",
        "src/agent_committee/report.py",
        "src/agent_committee/templates",
        "pyproject.toml",
        "uv.lock",
        *V1_FROZEN_DATA,
    ]


def test_banking77_split_and_prereg_hashes_are_unchanged():
    dev_raw = (ROOT / "data" / "dev.jsonl").read_bytes()
    assert hashlib.sha256(dev_raw).hexdigest() == V1_DEV_SHA
    assert (ROOT / "data" / "test.sha256").read_text(encoding="utf-8").strip() == V1_TEST_SHA
    manifest = json.loads((ROOT / "data" / "split_manifest.json").read_text(encoding="utf-8"))
    assert (manifest["dev_sha256"], manifest["test_sha256"]) == (V1_DEV_SHA, V1_TEST_SHA)
    assert hashlib.sha256((ROOT / "prereg.yaml").read_bytes()).hexdigest() == V1_PREREG_SHA
    assert load_prereg(ROOT / "prereg.yaml", task="banking77").task == "banking77"


def test_banking77_split_is_never_rebuilt_and_nothing_is_downloaded():
    called = []
    s = Scenario(**{**BANKING77.__dict__, "loader": lambda: called.append(1) or []})
    with pytest.raises(FileExistsError):
        build_split(s, repo=ROOT)
    assert called == []


# ---------- a new scenario plugs in end to end ----------
PER_LABEL = 30  # exactly the default SplitConfig.min_cases_per_label


def write_cases(path: Path, cases: list[dict]) -> Path:
    path.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    return path


def demo_repo(tmp_path: Path, task: str = "demo") -> tuple[Path, Scenario]:
    rows = [
        {"id": f"{x}{i:02d}", "text": f"{x} text {i}", "label": x}
        for x in "abc"
        for i in range(PER_LABEL)
    ]
    cases_file = write_cases(tmp_path / "cases.jsonl", rows)
    s = Scenario.standard("demo", lambda: load_local(cases_file))
    prereg = yaml.safe_load((ROOT / "prereg.yaml").read_text(encoding="utf-8"))
    prereg["task"] = task
    (tmp_path / s.prereg_path).parent.mkdir(parents=True)
    (tmp_path / s.prereg_path).write_text(yaml.safe_dump(prereg), encoding="utf-8")
    return tmp_path, s


def test_a_local_dataset_builds_its_own_split_hashes_and_sealed_file(tmp_path):
    repo, s = demo_repo(tmp_path)
    assert (s.freeze_tag, s.data_dir, sealed_dir(s)) == (
        "arms-frozen-demo",
        "data/demo",
        "sealed/demo",
    )
    manifest = build_split(s, repo=repo)
    assert manifest["n_dev"] + manifest["n_test"] == 3 * PER_LABEL and manifest["n_labels"] == 3
    assert manifest["meets_min_test_n"] is False  # 23 test cases, prereg min_test_n is far above
    dev = load_dev(repo / s.data_dir)  # hash checked
    assert len(dev) == manifest["n_dev"]
    assert (repo / sealed_dir(s) / "test.jsonl").is_file()
    assert not (repo / "data" / "dev.jsonl").exists()  # Banking77's files are untouched
    assert frozen_paths(s)[0] == "scenarios/demo/prereg.yaml"
    assert f"{s.data_dir}/test.sha256" in frozen_paths(s)
    assert s.dev_report in frozen_paths(s)  # outside arms/, so frozen on its own
    with pytest.raises(FileExistsError):
        build_split(s, repo=repo)


def test_a_prereg_written_for_another_scenario_is_refused(tmp_path):
    repo, s = demo_repo(tmp_path, task="banking77")
    with pytest.raises(ValueError, match="prereg of scenario 'banking77', not 'demo'"):
        build_split(s, repo=repo)


def test_a_scenario_without_arms_config_cannot_be_unsealed_or_run(tmp_path, monkeypatch, capsys):
    repo, s = demo_repo(tmp_path)
    build_split(s, repo=repo)
    with pytest.raises(SealError, match="no arms config"):
        check_unseal_conditions(repo, s)
    monkeypatch.setitem(scenarios.SCENARIOS, "demo", s)
    for extra in ([], ["--rehearse"]):
        assert runner.main(["--repo", str(repo), "--scenario", "demo", *extra]) == 1
        assert "no arms config" in capsys.readouterr().err


def test_unknown_scenario_is_refused():
    with pytest.raises(ValueError, match="unknown scenario"):
        get_scenario("nope")
    for bad in ("Bad-Name", "ünïcode", ""):
        with pytest.raises(ValueError, match="lowercase"):
            Scenario.standard(bad, list)
    with pytest.raises(ValueError, match="reserved"):
        Scenario.standard("rehearsal", list)


# ---------- the local loader and the split's input checks ----------
def test_local_loader_reads_csv_and_needs_an_id_column(tmp_path):
    csv = tmp_path / "cases.csv"
    csv.write_text("id,text,label\n1,hello,a\n2,bye,b\n", encoding="utf-8-sig")  # Excel BOM
    assert load_local(csv) == [
        Case(id="1", text="hello", label="a"),
        Case(id="2", text="bye", label="b"),
    ]
    no_id = write_cases(tmp_path / "x.jsonl", [{"text": "t", "label": "a"}])
    with pytest.raises(ValueError, match=r"row 1: needs string columns \['id'\]"):
        load_local(no_id)
    with pytest.raises(ValueError, match="jsonl or .csv"):
        load_local(tmp_path / "x.txt")


def test_split_refuses_duplicate_ids_and_single_case_labels():
    many = [Case(id=f"a{i}", text="t", label="a") for i in range(4)]
    with pytest.raises(ValueError, match="duplicate case ids"):
        split_cases([*many, many[0]], 0.25, seed=1)
    with pytest.raises(ValueError, match="fewer than 2 cases"):
        split_cases([*many, Case(id="b0", text="t", label="b")], 0.25, seed=1)


# ---------- scorers by name ----------
def test_scorers_resolve_by_name_only_from_the_registry():
    assert get_scorer("exact_label") is exact_label
    assert resolve_scorer("agent_committee.scorers:exact_label") is exact_label
    for bad in ("agent_committee.scorers:get_scorer", "os:system"):
        with pytest.raises(ValueError):
            resolve_scorer(bad)
    with pytest.raises(ValueError, match="unknown scorer"):
        get_scorer("llm_judge")


# ---------- the Hugging Face loader (no network) ----------
def test_huggingface_loader_pools_splits_with_stable_ids_and_label_names(monkeypatch):
    import sys
    import types

    class Split(list):
        def __init__(self, rows, names=None):
            super().__init__(rows)
            self.features = {"intent": types.SimpleNamespace(names=names) if names else object()}

    fake = {
        "train": Split(
            [{"q": "hi", "intent": 1}, {"q": "yo", "intent": 0}], names=["bye", "greet"]
        ),
        "test": Split([{"q": "ciao", "intent": 1}], names=["bye", "greet"]),
        "raw": Split([{"q": "x", "intent": "free text label"}]),
    }
    seen = []
    module = types.SimpleNamespace(load_dataset=lambda *a: seen.append(a) or fake)
    monkeypatch.setitem(sys.modules, "datasets", module)
    from agent_committee.data import load_huggingface

    got = load_huggingface(
        "org/ds", text="q", label="intent", splits=("train", "test"), id_prefix="d", config="plus"
    )
    assert seen == [("org/ds", "plus")]
    assert got == [
        Case(id="d-train-00000", text="hi", label="greet"),
        Case(id="d-train-00001", text="yo", label="bye"),
        Case(id="d-test-00000", text="ciao", label="greet"),
    ]
    raw = load_huggingface("org/ds", text="q", label="intent", splits=("raw",), id_prefix="r")
    assert raw == [Case(id="r-raw-00000", text="x", label="free text label")]


def test_the_serializer_reproduces_the_v1_dev_bytes():
    """Round trip through Case and to_jsonl_bytes: guards schema changes to Case (P4 `group`)."""
    from agent_committee.data import to_jsonl_bytes

    dev = load_dev(ROOT / "data")
    assert hashlib.sha256(to_jsonl_bytes(dev)).hexdigest() == V1_DEV_SHA


# ---------- P4: data handling for real teams ----------
def test_a_scenario_at_the_minimum_per_label_survives_the_arms_split(tmp_path):
    from agent_committee.arms.splits import make_arms_split

    repo, s = demo_repo(tmp_path)
    build_split(s, repo=repo)
    split = make_arms_split(load_dev(repo / s.data_dir))
    assert {c.label for c in split.pool} == {"a", "b", "c"}
    assert split.tune and split.eval


def test_too_few_cases_per_label_is_refused():
    cases = [Case(id=f"{x}{i}", text="t", label=x) for x in "ab" for i in range(29)]
    with pytest.raises(ValueError, match="fewer than 30 cases"):
        split_cases(cases, 0.25, seed=1, min_per_label=30)


def grouped_cases(n_groups: int = 12, per_group: int = 3) -> list[Case]:
    return [
        Case(id=f"{x}{g}-{i}", text=f"{x} {g} {i}", label=x, group=f"{x}g{g}")
        for x in "ab"
        for g in range(n_groups)
        for i in range(per_group)
    ]


def test_grouped_split_keeps_every_group_on_one_side_and_is_deterministic():
    cases = grouped_cases()
    dev, test = split_cases(cases, 0.25, seed=7, grouped=True, min_per_label=2)
    assert not {c.group for c in dev} & {c.group for c in test}
    assert {c.label for c in test} <= {c.label for c in dev}
    shuffled = list(reversed(cases))
    assert split_cases(shuffled, 0.25, seed=7, grouped=True, min_per_label=2) == (dev, test)


def test_grouped_manifest_records_the_realised_fraction_and_groups(tmp_path):
    from agent_committee.data import write_split

    dev, test = split_cases(grouped_cases(), 0.25, seed=7, grouped=True, min_per_label=2)
    m = write_split(
        dev, test, seed=7, test_fraction=0.25, data_dir=tmp_path / "d", sealed_dir=tmp_path / "s"
    )
    assert m["grouped"] is True and m["n_groups_shared"] == 0
    assert m["n_groups_dev"] + m["n_groups_test"] == 24
    assert m["realised_test_fraction"] == round(len(test) / 72, 4)
    assert load_dev(tmp_path / "d") == dev  # the group survives the round trip


def test_groups_need_the_grouped_setting_and_grouped_needs_groups():
    cases = grouped_cases()
    with pytest.raises(ValueError, match="set split.grouped explicitly"):
        split_cases(cases, 0.25, seed=1, min_per_label=2)
    plain = [Case(id=c.id, text=c.text, label=c.label) for c in cases]
    with pytest.raises(ValueError, match="every case needs a group"):
        split_cases(plain, 0.25, seed=1, grouped=True, min_per_label=2)
    one_group = [c.model_copy(update={"group": f"{c.label}-only"}) for c in cases]
    with pytest.raises(ValueError, match="fewer than 2 groups"):
        split_cases(one_group, 0.25, seed=1, grouped=True, min_per_label=2)


def test_local_loader_reads_an_optional_group_column(tmp_path):
    csv = tmp_path / "g.csv"
    csv.write_text("id,text,label,group\n1,hi,a,u1\n2,yo,a,\n", encoding="utf-8")
    assert [c.group for c in load_local(csv)] == ["u1", None]


def set_split(repo: Path, s: Scenario, **split) -> None:
    path = repo / s.prereg_path
    prereg = yaml.safe_load(path.read_text(encoding="utf-8"))
    prereg["split"].update(split)
    path.write_text(yaml.safe_dump(prereg), encoding="utf-8")


def test_build_split_applies_the_prereg_minimum_per_label(tmp_path):
    repo, s = demo_repo(tmp_path)
    set_split(repo, s, min_cases_per_label=PER_LABEL + 1)
    with pytest.raises(ValueError, match=f"fewer than {PER_LABEL + 1} cases"):
        build_split(s, repo=repo)


def test_build_split_applies_the_prereg_grouped_setting(tmp_path):
    rows = [
        {"id": f"{x}{g}-{i}", "text": f"{x} {g} {i}", "label": x, "group": f"{x}g{g}"}
        for x in "ab"
        for g in range(12)
        for i in range(3)
    ]
    cases_file = write_cases(tmp_path / "grouped.jsonl", rows)
    repo, _ = demo_repo(tmp_path)
    s = Scenario.standard("demo", lambda: load_local(cases_file))
    with pytest.raises(ValueError, match="set split.grouped explicitly"):
        build_split(s, repo=repo)
    set_split(repo, s, grouped=True)
    m = build_split(s, repo=repo)
    assert m["grouped"] is True and m["n_groups_shared"] == 0


def test_banking77_layer_rebuilds_to_the_v1_fingerprint():
    """The v1 linear layer, rebuilt from the committed DEV pool (reads DEV only)."""
    from agent_committee.seal import rebuild_fingerprint

    assert rebuild_fingerprint(ROOT) == V1_LAYER_FINGERPRINT
