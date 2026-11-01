"""Banking77 loading and the deterministic stratified DEV/TEST split.

Banking77 ships with train/test; we POOL them and re-split so that DEV and TEST are drawn from one
distribution under our own seed (recorded in prereg.yaml).

Sealing rules (see CLAUDE.md):
- This module WRITES the sealed TEST file once, at split creation, and never returns TEST rows.
- `load_dev` is the only loader exposed to arm-building code.
- Reading TEST back is `seal.load_test`'s job, and only the Phase D runner may call it.
"""

import hashlib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from agent_committee.schemas import Case

DEV_FILE = "dev.jsonl"
DEV_HASH_FILE = "dev.sha256"
TEST_HASH_FILE = "test.sha256"
MANIFEST_FILE = "split_manifest.json"
SEALED_TEST_FILE = "test.jsonl"


def to_jsonl_bytes(cases: list[Case]) -> bytes:
    """Canonical serialization: sorted keys, UTF-8, LF line endings. This is what gets hashed."""
    lines = (json.dumps(c.model_dump(), sort_keys=True, ensure_ascii=False) for c in cases)
    return ("\n".join(lines) + "\n").encode("utf-8")


def from_jsonl_bytes(raw: bytes) -> list[Case]:
    return [Case.model_validate_json(line) for line in raw.decode("utf-8").splitlines() if line]


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def split_cases(
    cases: list[Case], test_fraction: float, seed: int
) -> tuple[list[Case], list[Case]]:
    """Deterministic stratified split by label. Returns (dev, test), each sorted by case id."""
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


def main() -> None:
    """Build the split from prereg.yaml (run once: `python -m agent_committee.data`)."""
    from agent_committee.prereg import load_prereg

    prereg = load_prereg()
    dev, test = split_cases(load_banking77(), prereg.split.test_fraction, prereg.split.seed)
    manifest = write_split(
        dev, test, seed=prereg.split.seed, test_fraction=prereg.split.test_fraction
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
