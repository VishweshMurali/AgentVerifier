"""Saved free models for quick tests (owner approved 2026-11-12).

The free models (and the shared linear layer) are fitted on a scenario's training pool with fixed
settings and seeds, so fitting them again gives the same model. Quick tests therefore keep each
fitted model on disk and load it next time instead of refitting (3-8 minutes saved per test).

A saved model is only reused when everything it depends on is byte-identical: the training
messages and labels themselves (not the dataset's name), the arms' source code and the
scikit-learn version. Anything else is a new key and a fresh fit. Files are never deleted here;
a file that will not load is refitted and replaced. The sealed runner never uses this.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from agent_committee.schemas import Case

CACHE_DIR = Path("cache") / "quick-models"
ARMS_DIR = Path(__file__).parent / "arms"


def code_digest() -> str:
    """The arms' source code and the scikit-learn version: a change to either is a new key."""
    import sklearn

    h = hashlib.sha256(f"sklearn {sklearn.__version__}\n".encode())
    for f in sorted(ARMS_DIR.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def data_digest(*splits: Sequence[Case]) -> str:
    """The training messages themselves, in a fixed order (id, text, label)."""
    h = hashlib.sha256()
    for split in splits:
        h.update(b"--split--\n")
        for c in sorted(split, key=lambda c: c.id):
            # one JSON array per row: no text can be mistaken for a field or row boundary
            h.update(json.dumps([c.id, c.text, c.label], ensure_ascii=False).encode() + b"\n")
    return h.hexdigest()


def model_key(name: str, data: str, code: str) -> str:
    return hashlib.sha256(f"{name}|{data}|{code}".encode()).hexdigest()[:24]


def _write_atomic(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(blob)
    os.replace(tmp, path)


def load_or_fit(repo: Path, name: str, key: str, fit: Callable[[], Any]) -> tuple[Any, float, bool]:
    """(model, seconds the fit took, whether it was loaded). The fit's measured time is saved
    beside the model, so a loaded model still reports how long a fresh fit takes."""
    folder = repo / CACHE_DIR
    model_file, meta_file = folder / f"{name}-{key}.pkl", folder / f"{name}-{key}.json"
    if model_file.is_file() and meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            with model_file.open("rb") as f:
                return pickle.load(f), float(meta["fit_seconds"]), True  # noqa: S301 (own cache)
        except Exception:  # noqa: BLE001 (a broken file is simply refitted)
            pass
    t0 = time.perf_counter()
    model = fit()
    took = time.perf_counter() - t0
    _write_atomic(model_file, pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    _write_atomic(meta_file, json.dumps({"name": name, "fit_seconds": round(took, 2)}).encode())
    return model, took, False


def saved_fit_seconds(repo: Path, keys: Iterable[tuple[str, str]]) -> dict[str, float | None]:
    """For the time estimate: each model's saved fit time, or None when it is not saved yet."""
    out: dict[str, float | None] = {}
    for name, key in keys:
        meta = repo / CACHE_DIR / f"{name}-{key}.json"
        try:
            out[name] = float(json.loads(meta.read_text(encoding="utf-8"))["fit_seconds"])
        except (OSError, ValueError, KeyError):
            out[name] = None
    return out


def write_summary(
    repo: Path, scenario: str, code: str, fit_seconds: dict[str, float], **extra: Any
) -> None:
    """What the page's time estimate reads: fit times, the code they were saved with, and any
    measured speeds (e.g. the agent's seconds per answer). Merged into the scenario's file."""
    path = repo / CACHE_DIR / f"summary-{scenario}.json"
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    if old.get("code") != code:
        old = {}
    fits = {**old.get("fit_seconds", {}), **{k: round(v, 2) for k, v in fit_seconds.items()}}
    _write_atomic(path, json.dumps({**old, **extra, "code": code, "fit_seconds": fits}).encode())


def read_summary(repo: Path, scenario: str) -> dict[str, Any] | None:
    """The scenario's saved summary if it was written with the current code, else None."""
    try:
        out = json.loads((repo / CACHE_DIR / f"summary-{scenario}.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return out if out.get("code") == code_digest() else None
