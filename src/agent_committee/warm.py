"""Train the free models ahead of time (owner approved 2026-11-14), so a use case's first quick
test is as fast as a repeat.

For each ready scenario (its DEV split is built): the shared linear layer and every free model
are loaded-or-fitted with exactly the quick test's code and keys (`quick.load_or_fit_layer`,
`quick.load_or_fit_free`), so the saved models are the ones a test would have trained itself.
Models that are already saved are skipped without loading them. The local server starts this
at launch and after an upload; it never runs a test and never opens the sealed TEST.

    uv run python -m agent_committee.warm --all
    uv run python -m agent_committee.warm --scenario emotion
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from agent_committee import model_cache


def ready_scenarios(repo: Path) -> list[str]:
    """Scenarios whose DEV split exists (Banking77 plus every scenarios/<name> with data)."""
    from agent_committee.scenarios import get_scenario

    names = [
        "banking77",
        *sorted(p.parent.name for p in (repo / "scenarios").glob("*/scenario.yaml")),
    ]
    out = []
    for name in names:
        try:
            if (repo / get_scenario(name, repo).data_dir / "dev.jsonl").is_file():
                out.append(name)
        except Exception:  # noqa: BLE001 (a broken scenario is skipped, not fatal)
            continue
    return out


def warm(repo: Path, scenario: str, log=print, only=None) -> dict[str, float]:
    """Fit and save whatever is missing for one scenario; returns each model's fit seconds.
    `only` names the models to prepare (the shared layer always is); None is every free model."""
    from agent_committee import quick
    from agent_committee.arms.evaluate import CLASSIC
    from agent_committee.arms.splits import make_arms_split
    from agent_committee.data import load_dev
    from agent_committee.scenarios import get_scenario

    s = get_scenario(scenario, repo)
    split = make_arms_split(load_dev(repo / s.data_dir))
    layer, layer_seconds, _ = quick.load_or_fit_layer(repo, split)
    seconds = {"linear": layer_seconds}
    for name in CLASSIC:
        if name == "linear" or (only is not None and name not in only):
            continue
        key = quick.free_model_key(name, split)
        saved = repo / model_cache.CACHE_DIR / f"{name}-{key}.pkl"
        if saved.is_file():  # already saved: skip without loading it (some are GBs)
            log(f"{scenario}: {name} already saved")
            continue
        t0 = time.perf_counter()
        _, seconds[name], loaded = quick.load_or_fit_free(repo, name, split, layer)
        took = time.perf_counter() - t0
        log(f"{scenario}: {name} {'loaded' if loaded else 'trained'} in {took:.0f} s")
    model_cache.write_summary(repo, scenario, model_cache.code_digest(), seconds)
    return seconds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the free models ahead of time.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario")
    group.add_argument("--all", action="store_true")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--arms", help="comma-separated models to prepare (default: every free one)")
    args = ap.parse_args(argv)
    only = {a.strip() for a in args.arms.split(",") if a.strip()} if args.arms else None
    repo = Path(args.repo)
    names = ready_scenarios(repo) if args.all else [args.scenario]
    failed = 0
    for name in names:
        try:
            warm(repo, name, only=only)
        except Exception as exc:  # noqa: BLE001 (one scenario failing must not stop the rest)
            failed += 1
            print(f"{name}: could not prepare ({exc})", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
