"""Background jobs. Only one exists: the DEV rehearsal (`runner --rehearse`, never opens TEST).

The sealed run is deliberately not offered here: it runs once, by hand, after the user's OK.
One job at a time; output is kept as a short tail.
"""

import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REHEARSAL_CMD = ["uv", "run", "python", "-m", "agent_committee.runner", "--rehearse"]


@dataclass
class Job:
    id: str
    kind: str
    state: str = "running"  # running | succeeded | failed
    started: float = field(default_factory=time.time)
    finished: float | None = None
    returncode: int | None = None
    tail: deque[str] = field(default_factory=lambda: deque(maxlen=40))

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "started": self.started,
            "finished": self.finished,
            "returncode": self.returncode,
            "output_tail": list(self.tail),
        }


class JobBusy(RuntimeError):
    pass


class JobManager:
    def __init__(self, root: Path, cmd: list[str] | None = None):
        self.root = root
        self.cmd = cmd or REHEARSAL_CMD
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def running(self) -> Job | None:
        return next((j for j in self._jobs.values() if j.state == "running"), None)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def start_rehearsal(self) -> Job:
        with self._lock:
            if busy := self.running():
                raise JobBusy(busy.id)
            job = Job(id=uuid.uuid4().hex[:12], kind="rehearsal")
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job) -> None:
        try:
            proc = subprocess.Popen(
                self.cmd,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                job.tail.append(line.rstrip())
            job.returncode = proc.wait()
            job.state = "succeeded" if job.returncode == 0 else "failed"
        except OSError as exc:
            job.tail.append(f"could not start: {exc}")
            job.state = "failed"
        job.finished = time.time()
