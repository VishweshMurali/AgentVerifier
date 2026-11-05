"""Live status of the project: Ollama, the sealed run, the rehearsal, the run-phase progress.

Reads only file metadata, the split manifest and row COUNTS from the LLM cache. It never opens the
sealed test file and never reads a prompt or an answer from the cache.
"""

import json
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OLLAMA_TAGS = "http://localhost:11434/api/tags"
RUN_MARKER = "sealed_run.started"  # same name as agent_committee.runner.RUN_MARKER


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def sealed_verdict(self) -> Path:
        return self.results / "verdict.json"

    @property
    def rehearsal_verdict(self) -> Path:
        return self.results / "rehearsal" / "verdict.json"

    @property
    def marker(self) -> Path:
        return self.results / RUN_MARKER

    @property
    def cache(self) -> Path:
        return self.root / "cache" / "llm.sqlite"

    @property
    def manifest(self) -> Path:
        return self.root / "data" / "split_manifest.json"

    @property
    def dist(self) -> Path:
        return self.root / "frontend" / "dist-server"


def ollama_up(url: str = OLLAMA_TAGS, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as res:  # noqa: S310 (fixed local URL)
            return 200 <= res.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def run_progress(cache: Path) -> dict[str, int]:
    """Run-phase answers cached so far, per arm. Counts only; no row content is read."""
    if not cache.exists():
        return {}
    con = sqlite3.connect(f"file:{cache.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT arm, COUNT(*) FROM cache WHERE phase = 'run' GROUP BY arm"
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        con.close()
    return {str(arm): int(n) for arm, n in rows}


def _file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return {"mtime": path.stat().st_mtime}


def project_status(paths: Paths, ollama: bool) -> dict[str, Any]:
    manifest = (
        json.loads(paths.manifest.read_text(encoding="utf-8")) if paths.manifest.exists() else {}
    )
    sealed = _file(paths.sealed_verdict)
    started = paths.marker.exists()
    if sealed:
        state = "finished"
    elif started:
        state = "started_not_finished"
    else:
        state = "not_started"
    return {
        "ollama": ollama,
        "sealed_run": {
            "state": state,
            "started_at_commit": (
                paths.marker.read_text(encoding="utf-8").strip() if started else None
            ),
            "verdict": sealed,
            # test cases before copies of DEV text are dropped (the kept count is in the verdict)
            "test_cases": manifest.get("n_test"),
            "answers_cached": run_progress(paths.cache),
        },
        "rehearsal": {"verdict": _file(paths.rehearsal_verdict)},
    }
