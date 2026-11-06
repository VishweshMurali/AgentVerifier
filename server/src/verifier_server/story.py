"""The page's story, built by the same code as the static export (scripts/export_frontend_data.py).

The sealed verdict wins over the rehearsal. The prereg must be the one the verdict was judged
under (`load_checked_prereg`), or the story is refused. The one replayed case comes from
`<root>/cache/llm.sqlite`, opened read-only by the export (no model calls, nothing written).

Caching: the story is rebuilt when the verdict file changes. The case does not depend on the
verdict, so it is replayed again only when the cache file has changed too. The replay (slow: it
rebuilds the linear layer) runs under its own lock, never under the story lock.
"""

import importlib.util
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from .status import Paths

_UNSET = object()


def load_export(root: Path) -> ModuleType:
    path = root / "scripts" / "export_frontend_data.py"
    spec = importlib.util.spec_from_file_location("export_frontend_data", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerdictUnreadable(RuntimeError):
    """The verdict cannot be shown: the file does not parse (being written, or broken), or the
    export refused it (prereg mismatch, missing default price table, rehearsal shown as sealed)."""


class StoryService:
    def __init__(self, paths: Paths, with_case: bool = True, script_root: Path | None = None):
        self.paths = paths
        self.with_case = with_case
        self._export = load_export(script_root or paths.root)
        self._lock = threading.Lock()  # guards _key/_story; held only for quick work
        self._case_lock = threading.Lock()  # serialises the slow replay
        self._key: tuple[str, int] | None = None
        self._story: dict[str, Any] | None = None
        self._case_key: int | None = None
        self._case: Any = _UNSET

    def get(self) -> dict[str, Any]:
        """Raises FileNotFoundError when there is no verdict yet, VerdictUnreadable when it cannot
        be shown (the runner writes the verdict in place, so a read can land mid-write)."""
        try:
            verdict, source = self._export.pick_verdict(
                self.paths.sealed_verdict, self.paths.rehearsal_verdict
            )
        except ValueError as exc:  # pydantic/JSON errors and ExportRefused
            raise VerdictUnreadable(str(exc)) from exc
        path = self.paths.sealed_verdict if source == "sealed" else self.paths.rehearsal_verdict
        key = (source, path.stat().st_mtime_ns)
        with self._lock:
            if self._key == key and self._story is not None:
                return self._story
        case = self._replayed_case()  # outside the story lock
        try:
            prereg = self._export.load_checked_prereg(verdict, self.paths.root / "prereg.yaml")
            story = self._export.build_story(verdict, source, case, prereg)
        except ValueError as exc:  # ExportRefused
            raise VerdictUnreadable(str(exc)) from exc
        with self._lock:
            self._key, self._story = key, story
        return story

    def _replayed_case(self) -> dict[str, Any] | None:
        if not self.with_case:
            return None
        try:
            cache_key: int | None = self.paths.cache.stat().st_mtime_ns
        except OSError:
            cache_key = None
        with self._case_lock:
            if self._case is _UNSET or self._case_key != cache_key:
                self._case = self._export.replay_case(self.paths.root)
                self._case_key = cache_key
            return self._case
