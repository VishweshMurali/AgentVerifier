"""Background jobs: the DEV rehearsal (`runner --rehearse`) and quick tests
(`agent_committee.quick`), both on DEV only; neither opens TEST.

The sealed run is deliberately not offered here: it runs once, by hand, after the user's OK.
One job at a time, also across server restarts: a pid file (results/rehearsal/job.pid) records the
running child, and a new rehearsal is refused while that process is still alive. The child is
killed when the server shuts down or when it exceeds the timeout. Output is kept as a short tail.
"""

import ctypes
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REHEARSAL_CMD = ["uv", "run", "python", "-m", "agent_committee.runner", "--rehearse"]
TIMEOUT_S = 12 * 3600.0  # a DEV rehearsal on a local 8B model takes hours, never half a day
PID_FILE = Path("results") / "rehearsal" / "job.pid"


@dataclass
class Job:
    id: str
    kind: str
    state: str = "running"  # running | succeeded | failed
    started: float = field(default_factory=time.time)
    finished: float | None = None
    returncode: int | None = None
    tail: deque[str] = field(default_factory=lambda: deque(maxlen=40))
    # parsed from the output as it arrives (quick tests); kept whole, unlike the tail
    progress: dict[str, Any] = field(default_factory=dict)
    cmd: list[str] = field(default_factory=list, repr=False)
    timeout_s: float = TIMEOUT_S
    parse: Callable[[dict[str, Any], str], None] | None = field(default=None, repr=False)

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "started": self.started,
            "finished": self.finished,
            "returncode": self.returncode,
            "output_tail": list(self.tail),
            "progress": self.progress,
        }


class JobBusy(RuntimeError):
    pass


def pid_alive(pid: int) -> bool:
    """True while a process with this pid exists. Never signals or stops it."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class JobManager:
    def __init__(self, root: Path, cmd: list[str] | None = None, timeout_s: float = TIMEOUT_S):
        self.root = root
        self.cmd = cmd or REHEARSAL_CMD
        self.timeout_s = timeout_s
        self.pid_file = root / PID_FILE
        self._jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def running(self) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs.values() if j.state == "running"), None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def orphan_pid(self) -> int | None:
        """The pid of a rehearsal left by an earlier server process, if it is still alive.

        A pid file older than the timeout is stale (the pid may have been reused) and is ignored.
        """
        try:
            text = self.pid_file.read_text(encoding="utf-8").strip()
            age = time.time() - self.pid_file.stat().st_mtime
            pid = int(text)
        except (OSError, ValueError):
            return None
        if age > self.timeout_s or not pid_alive(pid):
            return None
        return pid

    def start_rehearsal(self) -> Job:
        return self.start("rehearsal", self.cmd, self.timeout_s)

    def start(
        self,
        kind: str,
        cmd: list[str],
        timeout_s: float,
        parse: Callable[[dict[str, Any], str], None] | None = None,
    ) -> Job:
        """Start one job; refused while any job (or an earlier server's child) is running."""
        with self._lock:
            if busy := next((j for j in self._jobs.values() if j.state == "running"), None):
                raise JobBusy(busy.id)
            if (pid := self.orphan_pid()) is not None:
                raise JobBusy(f"process {pid} from an earlier server is still running")
            job = Job(
                id=uuid.uuid4().hex[:12], kind=kind, cmd=list(cmd), timeout_s=timeout_s, parse=parse
            )
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def shutdown(self) -> None:
        """Kill every running child (called when the server stops)."""
        with self._lock:
            procs = list(self._procs.values())
        for proc in procs:
            if proc.poll() is None:
                proc.kill()

    def _run(self, job: Job) -> None:
        timer: threading.Timer | None = None
        timed_out = threading.Event()
        try:
            proc = subprocess.Popen(
                job.cmd,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            with self._lock:
                self._procs[job.id] = proc
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            self.pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")

            def kill_on_timeout() -> None:
                timed_out.set()
                proc.kill()

            timer = threading.Timer(job.timeout_s, kill_on_timeout)
            timer.daemon = True
            timer.start()
            assert proc.stdout is not None
            for line in proc.stdout:
                job.tail.append(line.rstrip())
                if job.parse is not None:
                    try:
                        job.parse(job.progress, line)
                    except Exception as exc:  # a parsing slip must not end the run
                        job.tail.append(f"progress parse error: {exc!r}")
            job.returncode = proc.wait()
            if timed_out.is_set():
                job.tail.append(f"stopped: exceeded the {job.timeout_s:.0f} s timeout")
            ok = job.returncode == 0 and not timed_out.is_set()
            job.state = "succeeded" if ok else "failed"
        except Exception as exc:  # any failure must end the job, or it blocks every later one
            job.tail.append(f"job error: {exc!r}")
            job.state = "failed"
        finally:
            if timer is not None:
                timer.cancel()
            with self._lock:
                self._procs.pop(job.id, None)
            self.pid_file.unlink(missing_ok=True)
            job.finished = time.time()
            if job.state == "running":  # e.g. a BaseException escaped
                job.state = "failed"
