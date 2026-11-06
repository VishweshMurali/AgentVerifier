"""Check everything with one command: PASS or FAIL per step, non-zero exit on any failure.

    uv run python scripts/check.py            # everything (installs front-end packages with npm ci)
    uv run python scripts/check.py --no-install   # reuse frontend/node_modules
    uv run python scripts/check.py --fast         # skip the front-end build and the end-to-end test

No step calls a model, reads the LLM cache or opens sealed/: the tests use fake transports.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def npm() -> str:
    exe = shutil.which("npm") or shutil.which("npm.cmd")
    if not exe:
        sys.exit("npm not found on PATH")
    return exe


def node_modules_in_use(fe: Path) -> bool:
    """True when another process holds frontend/node_modules open (e.g. a dev or preview server).

    `npm ci` deletes node_modules first; on Windows a held file makes it fail halfway and leaves
    the folder broken. Renaming the folder and back is a cheap probe: it fails while files are open.
    """
    nm = fe / "node_modules"
    if not nm.is_dir():
        return False
    probe = fe / "node_modules.check-probe"
    try:
        nm.rename(probe)
    except OSError:
        return True
    probe.rename(nm)
    return False


def steps(install: bool, fast: bool) -> list[tuple[str, list[str], Path]]:
    fe = ROOT / "frontend"
    out = [
        ("pipeline tests", ["uv", "run", "pytest", "-q"], ROOT),
        ("lint", ["uv", "run", "ruff", "check", "."], ROOT),
        ("format", ["uv", "run", "ruff", "format", "--check", "."], ROOT),
        (
            "server tests",
            ["uv", "run", "--project", "server", "pytest", "-q", "server/tests"],
            ROOT,
        ),
    ]
    if install:
        out.append(("front-end install", [npm(), "ci"], fe))
    out.append(("front-end tests", [npm(), "test"], fe))
    if not fast:
        out += [
            ("front-end build", [npm(), "run", "build"], fe),
            ("server page build", [npm(), "run", "build:server"], fe),
            (
                "end-to-end",
                ["uv", "run", "--project", "server", "pytest", "-q", "server/e2e"],
                ROOT,
            ),
        ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-install", action="store_true", help="skip npm ci")
    ap.add_argument("--fast", action="store_true", help="skip builds and the end-to-end test")
    args = ap.parse_args()
    if not args.no_install and node_modules_in_use(ROOT / "frontend"):
        print(
            "frontend/node_modules is in use (a dev or preview server is probably running).\n"
            "Stop it, or run with --no-install; npm ci would leave node_modules broken."
        )
        return 1
    results: list[tuple[str, bool, float]] = []
    for name, cmd, cwd in steps(not args.no_install, args.fast):
        shown = " ".join([Path(cmd[0]).stem, *cmd[1:]])
        print(f"\n=== {name}: {shown}")
        t0 = time.monotonic()
        ok = subprocess.run(cmd, cwd=cwd).returncode == 0
        results.append((name, ok, time.monotonic() - t0))
        if not ok:
            break
    print("\n=== summary")
    for name, ok, secs in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({secs:.0f}s)")
    failed = [n for n, ok, _ in results if not ok]
    print("\nALL PASSED" if not failed else f"\nFAILED: {failed[0]} (later steps not run)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
