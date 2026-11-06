"""Print required N (prereg power) for the prereg min_gap across discordance rates.

    uv run python scripts/power_analysis.py [--sims 1000]

Also prints the simulated power of the full DEPLOY_AGENT rule at the headline discordance.
"""

import argparse
import json
from pathlib import Path

from agent_committee.power import (
    DEPLOY_SIMS,
    HEADLINE_DISCORDANCE,
    deploy_rule_power,
    mcnemar_power,
    power_table,
)
from agent_committee.prereg import load_prereg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=DEPLOY_SIMS, help="runs for the full-rule power")
    args = ap.parse_args(argv)
    prereg = load_prereg("prereg.yaml")
    rows = power_table(prereg.min_gap, prereg.alpha, prereg.power)
    manifest = json.loads(Path("data/split_manifest.json").read_text(encoding="utf-8"))
    n_test = manifest["n_test"]
    print(
        f"min_gap={prereg.min_gap} alpha={prereg.alpha} target power={prereg.power} "
        "(exact two-sided McNemar; min_gap is ASSUMED)"
    )
    print(f"{'discordance':>12} {'required N':>11} {'power@N_test':>13}")
    headline_n = None
    for r in rows:
        p = mcnemar_power(n_test, prereg.min_gap, r.discordance, prereg.alpha)
        mark = ""
        if r.discordance == HEADLINE_DISCORDANCE:
            mark, headline_n = "  <- headline", r.required_n
        print(f"{r.discordance:>12.2f} {r.required_n:>11,} {p:>13.3f}{mark}")
    print(f"N_test = {n_test:,}")
    print(
        f"\nFull DEPLOY_AGENT rule (McNemar p < alpha, CI lower bound > 0, gain >= min_gap), "
        f"simulated at a true gap of min_gap, discordance {HEADLINE_DISCORDANCE}, "
        f"{args.sims} runs, {prereg.stats.bootstrap_replicates} bootstrap resamples, fixed seed:"
    )
    for label, n in (("required N", headline_n), ("N_test", n_test)):
        if n is None:
            continue
        mc = mcnemar_power(n, prereg.min_gap, HEADLINE_DISCORDANCE, prereg.alpha)
        full = deploy_rule_power(
            n,
            prereg.min_gap,
            HEADLINE_DISCORDANCE,
            prereg.alpha,
            prereg.min_gap,
            prereg.stats.ci_level,
            sims=args.sims,
            replicates=prereg.stats.bootstrap_replicates,
        )
        print(f"  {label} = {n:,}: McNemar power {mc:.3f}, full-rule power {full:.3f}")
    print(
        "NOTE: required N powers the McNemar test only. The full DEPLOY_AGENT rule has LOWER\n"
        "power: when the true gap equals min_gap, the observed gain falls below min_gap about\n"
        "half the time, so the agent is deployed in only about half of such runs. To reach the\n"
        "target power the true gap must be larger than min_gap. The true discordance is unknown\n"
        "until the arms are evaluated on DEV."
    )


if __name__ == "__main__":
    main()
