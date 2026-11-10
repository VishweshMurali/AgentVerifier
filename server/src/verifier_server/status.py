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

from agent_committee.llm import BASE_URL  # the one Ollama address setting

OLLAMA_TAGS = BASE_URL.rstrip("/").removesuffix("/v1") + "/api/tags"
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
    def prereg(self) -> Path:
        return self.root / "prereg.yaml"

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


def run_progress(cache: Path) -> tuple[dict[str, int], str | None]:
    """Run-phase answers cached so far, per arm, and the read error if the cache could not be read
    (e.g. locked by a writer), so "unreadable" never looks like "0 answers". Counts only; no row
    content is read."""
    if not cache.exists():
        return {}, None
    try:
        con = sqlite3.connect(f"file:{cache.as_posix()}?mode=ro", uri=True, timeout=0.5)
        try:
            rows = con.execute(
                "SELECT arm, COUNT(*) FROM cache WHERE phase = 'run' GROUP BY arm"
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:
        return {}, f"could not read the LLM cache: {exc}"
    return {str(arm): int(n) for arm, n in rows}, None


def _file(path: Path) -> dict[str, Any] | None:
    try:
        return {"mtime": path.stat().st_mtime}
    except OSError:
        return None


def _manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, or partly written
        return {}
    return data if isinstance(data, dict) else {}


def _marker_commit(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def project_status(paths: Paths, ollama: bool) -> dict[str, Any]:
    manifest = _manifest(paths.manifest)
    sealed = _file(paths.sealed_verdict)
    started = paths.marker.exists()
    if sealed:
        state = "finished"
    elif started:
        state = "started_not_finished"
    else:
        state = "not_started"
    answers, answers_error = run_progress(paths.cache)
    return {
        "ollama": ollama,
        "sealed_run": {
            "state": state,
            "started_at_commit": _marker_commit(paths.marker) if started else None,
            "verdict": sealed,
            # test cases before copies of DEV text are dropped (the kept count is in the verdict)
            "test_cases": manifest.get("n_test"),
            "answers_cached": answers,
            "answers_cached_error": answers_error,
        },
        "rehearsal": {"verdict": _file(paths.rehearsal_verdict)},
    }
