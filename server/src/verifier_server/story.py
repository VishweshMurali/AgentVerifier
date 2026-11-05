"""The page's story, built by the same code as the static export (scripts/export_frontend_data.py).

The sealed verdict wins over the rehearsal. The one replayed case comes from the build-phase LLM
cache only (no model calls) and is cached until the verdict file changes.
"""

import importlib.util
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from agent_committee.prereg import load_prereg

from .status import Paths


def load_export(root: Path) -> ModuleType:
    path = root / "scripts" / "export_frontend_data.py"
    spec = importlib.util.spec_from_file_location("export_frontend_data", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StoryService:
    def __init__(self, paths: Paths, with_case: bool = True, script_root: Path | None = None):
        self.paths = paths
        self.with_case = with_case
        self._export = load_export(script_root or paths.root)
        self._lock = threading.Lock()
        self._key: tuple[str, float] | None = None
        self._story: dict[str, Any] | None = None

    def get(self) -> dict[str, Any]:
        """Raises FileNotFoundError when there is no verdict yet."""
        verdict, source = self._export.pick_verdict(
            self.paths.sealed_verdict, self.paths.rehearsal_verdict
        )
        path = self.paths.sealed_verdict if source == "sealed" else self.paths.rehearsal_verdict
        key = (source, path.stat().st_mtime)
        with self._lock:
            if self._key != key or self._story is None:
                case = self._export.replay_case() if self.with_case else None
                prereg = load_prereg(self.paths.root / "prereg.yaml")
                self._story = self._export.build_story(verdict, source, case, prereg)
                self._key = key
            return self._story
