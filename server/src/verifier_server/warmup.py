"""Pre-training in the background (owner approved 2026-11-14): at launch and after an upload the
server runs `python -m agent_committee.warm` as a low-priority child process, one at a time, so
a use case's first quick test only waits for the agent. It never runs a test and never imports
the sealed-test code; a quick test that starts meanwhile simply loads or fits what it needs."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

BELOW_NORMAL = 0x00004000  # Windows: a lower CPU priority, so tests and the page stay quick


class Prewarmer:
    def __init__(self, root: Path, arms: tuple[str, ...] | None = None) -> None:
        self.root = root
        # a small host trains only the models it offers (None = every free model)
        self._only = ["--arms", ",".join(arms)] if arms else []
        self._lock = threading.Lock()
        self._queue: list[list[str]] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self.state: dict[str, Any] = {"running": None, "done": [], "failed": []}

    def request(self, scenario: str | None = None) -> None:
        """Prepare one scenario, or every ready one (None)."""
        args = ["--all"] if scenario is None else ["--scenario", scenario]
        with self._lock:
            if args in self._queue:
                return
            self._queue.append(args)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._drain, daemon=True)
                self._thread.start()

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self.state["running"] = None
                    return
                args = self._queue.pop(0)
                self.state["running"] = " ".join(args)
            flags = BELOW_NORMAL if sys.platform == "win32" else 0
            started = time.time()
            try:
                self._proc = subprocess.Popen(
                    [sys.executable, "-u", "-m", "agent_committee.warm", *args, *self._only],
                    cwd=self.root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=flags,
                )
                code = self._proc.wait()
            except OSError:
                code = -1
            entry = {"what": " ".join(args), "seconds": round(time.time() - started)}
            self.state["done" if code == 0 else "failed"].append(entry)

    def shutdown(self) -> None:
        with self._lock:
            self._queue.clear()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
