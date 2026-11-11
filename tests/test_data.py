import json
from collections import Counter
from pathlib import Path

import pytest

from agent_committee.data import (
    from_jsonl_bytes,
    load_dev,
    sha256_hex,
    split_cases,
    to_jsonl_bytes,
    write_split,
)
from agent_committee.schemas import Case


def _cases(n_labels=5, per_label=20):
    return [
        Case(id=f"c{k}-{i:03d}", text=f"text {k} {i}", label=f"L{k}")
        for k in range(n_labels)
        for i in range(per_label)
    ]


def test_split_is_deterministic_for_a_seed_and_differs_across_seeds():
    cases = _cases()
    a = split_cases(cases, 0.25, seed=1)
    assert a == split_cases(cases, 0.25, seed=1)
    assert a == split_cases(list(reversed(cases)), 0.25, seed=1)  # input order is irrelevant
    assert a != split_cases(cases, 0.25, seed=2)


def test_split_is_disjoint_complete_and_stratified():
    cases = _cases()
    dev, test = split_cases(cases, 0.25, seed=3)
    assert not {c.id for c in dev} & {c.id for c in test}
    assert len(dev) + len(test) == len(cases)
    assert Counter(c.label for c in test) == Counter({f"L{k}": 5 for k in range(5)})
    assert Counter(c.label for c in dev) == Counter({f"L{k}": 15 for k in range(5)})


def test_jsonl_roundtrip_is_byte_stable():
    cases = _cases(2, 3)
    raw = to_jsonl_bytes(cases)
    assert from_jsonl_bytes(raw) == cases
    assert to_jsonl_bytes(from_jsonl_bytes(raw)) == raw
    assert b"\r" not in raw


def test_write_split_writes_hashes_manifest_and_load_dev_roundtrips(tmp_path):
    dev, test = split_cases(_cases(), 0.25, seed=1)
    data_dir, sealed_dir = tmp_path / "data", tmp_path / "sealed"
    manifest = write_split(
        dev, test, seed=1, test_fraction=0.25, data_dir=data_dir, sealed_dir=sealed_dir
    )
    assert load_dev(data_dir) == dev
    assert (data_dir / "test.sha256").read_text().strip() == sha256_hex(
        (sealed_dir / "test.jsonl").read_bytes()
    )
    assert json.loads((data_dir / "split_manifest.json").read_text()) == manifest
    assert manifest["n_dev"] == len(dev) and manifest["n_test"] == len(test)
    # TEST text never lands in the committed data dir.
    assert not any("test.jsonl" in p.name for p in data_dir.iterdir())


def test_manifest_counts_test_texts_duplicated_in_dev(tmp_path):
    dev = [Case(id="d1", text="Hello  World", label="a")]
    test = [Case(id="t1", text="hello world", label="a"), Case(id="t2", text="other", label="a")]
    m = write_split(
        dev, test, seed=0, test_fraction=0.5, data_dir=tmp_path / "d", sealed_dir=tmp_path / "s"
    )
    assert m["n_test_texts_duplicated_in_dev"] == 1


@pytest.mark.parametrize(
    "existing",
    ["d/dev.jsonl", "d/dev.sha256", "d/test.sha256", "d/split_manifest.json", "s/test.jsonl"],
)
def test_write_split_refuses_to_overwrite_and_writes_nothing(tmp_path, existing):
    dev, test = split_cases(_cases(), 0.25, seed=1)
    (tmp_path / "d").mkdir()
    (tmp_path / "s").mkdir()
    pre = tmp_path / existing
    pre.write_text("precious")
    with pytest.raises(FileExistsError):
        write_split(
            dev,
            test,
            seed=1,
            test_fraction=0.25,
            data_dir=tmp_path / "d",
            sealed_dir=tmp_path / "s",
        )
    assert pre.read_text() == "precious"
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {existing}  # nothing else was created


def test_a_second_build_that_passed_the_existence_check_cannot_replace_the_split(
    tmp_path, monkeypatch
):
    """R8-D2: two builds both pass the up-front check; the later one must fail on its first file
    (the sealed TEST, created exclusively) and leave the first split, and its hash, untouched."""
    dev, test = split_cases(_cases(), 0.25, seed=1)
    kw = dict(seed=1, test_fraction=0.25, data_dir=tmp_path / "d", sealed_dir=tmp_path / "s")
    write_split(dev, test, **kw)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(Path, "exists", lambda self: False)  # the race: the check saw nothing
    other_dev, other_test = split_cases(_cases(), 0.25, seed=2)
    with pytest.raises(FileExistsError):
        write_split(other_dev, other_test, **kw)
    monkeypatch.undo()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_load_dev_detects_tampering(tmp_path):
    dev, test = split_cases(_cases(), 0.25, seed=1)
    write_split(
        dev, test, seed=1, test_fraction=0.25, data_dir=tmp_path / "d", sealed_dir=tmp_path / "s"
    )
    p = tmp_path / "d" / "dev.jsonl"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        load_dev(tmp_path / "d")
