"""The dataset menu: standard options, choosing one, preloading without side effects."""

import json

import pytest

from agent_committee import catalog
from agent_committee.scenarios import get_scenario


def test_menu_offers_the_checked_presets_and_your_own_dataset(tmp_path):
    menu = catalog.menu(tmp_path)
    assert [m["option"] for m in menu] == ["banking77", "clinc150", "emotion", "own"]
    by = {m["option"]: m for m in menu}
    assert by["banking77"]["status"] == "built in"
    assert (
        by["banking77"]["arms"] == catalog.BANKING77_ARMS and "used in v1" in catalog.BANKING77_ARMS
    )
    assert by["clinc150"]["status"] == "available"
    assert (
        by["clinc150"]["arms"] == catalog.NEW_DATASET_ARMS
        and "no rehearsal or verdict" in by["clinc150"]["arms"]
    )
    assert "cases" not in by["clinc150"]  # sizes only from a built manifest


def test_choosing_writes_the_scenario_file_once_and_reports_status(tmp_path):
    out = catalog.choose("clinc150", tmp_path)
    assert out["status"] == "chosen"
    s = get_scenario("clinc150", tmp_path)
    assert (
        s.definition == "scenarios/clinc150/scenario.yaml"
        and s.freeze_tag == "arms-frozen-clinc150"
    )
    assert catalog.choose("clinc150", tmp_path)["status"] == "chosen"  # again: status, no error
    assert catalog.choose("banking77", tmp_path)["status"] == "built in"
    assert not (tmp_path / "scenarios" / "banking77").exists()
    with pytest.raises(ValueError, match="unknown option"):
        catalog.choose("ag_news", tmp_path)


def test_status_shows_sizes_once_the_split_is_built(tmp_path):
    catalog.choose("emotion", tmp_path)
    data = tmp_path / "data" / "emotion"
    data.mkdir(parents=True)
    (data / "dev.jsonl").write_text("", encoding="utf-8")
    (data / "split_manifest.json").write_text(json.dumps({"n_dev": 15, "n_test": 5, "n_labels": 6}))
    emotion = {m["option"]: m for m in catalog.menu(tmp_path)}["emotion"]
    assert (emotion["status"], emotion["cases"], emotion["labels"]) == ("split built", 20, 6)


def test_preload_downloads_but_writes_no_scenario_and_never_raises(tmp_path, monkeypatch):
    calls = []

    def fake_loader(spec, repo):
        calls.append(spec.name)
        if spec.name == "emotion":
            return lambda: (_ for _ in ()).throw(ConnectionError("offline"))
        return lambda: [object()] * 3

    monkeypatch.setattr(catalog, "loader_for", fake_loader)
    out = catalog.preload(tmp_path)
    assert calls == ["clinc150", "emotion"]
    assert out["clinc150"] == "ready: 3 cases" and out["emotion"].startswith("failed:")
    assert not (tmp_path / "scenarios").exists()


def test_presets_use_no_remote_code_and_name_their_columns():
    for p in catalog.PRESETS.values():
        if p.spec is None:
            continue
        src = p.spec.source
        assert src.kind == "huggingface" and src.text and src.label and src.splits
        assert src.revision and len(src.revision) == 40  # pinned to a dataset commit
        assert p.spec.name == p.name


def test_a_scenario_of_a_preset_name_that_is_not_the_preset_shows_a_conflict(tmp_path):
    from agent_committee.scenarios import LocalSource, ScenarioFile, add_scenario

    own = ScenarioFile(name="clinc150", source=LocalSource(kind="local", path="sources/x.jsonl"))
    add_scenario(own, tmp_path)
    clinc = {m["option"]: m for m in catalog.menu(tmp_path)}["clinc150"]
    assert clinc["status"].startswith("conflict:")


def test_cli_prints_the_menu_and_chooses(tmp_path, capsys):
    assert catalog.main(["--repo", str(tmp_path)]) == 0
    assert "clinc150" in capsys.readouterr().out
    assert catalog.main(["--repo", str(tmp_path), "--choose", "emotion"]) == 0
    assert (tmp_path / "scenarios" / "emotion" / "scenario.yaml").is_file()
