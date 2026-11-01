"""Print required N (prereg power) for the prereg min_gap across discordance rates."""

import json
from pathlib import Path

from agent_committee.power import HEADLINE_DISCORDANCE, mcnemar_power, power_table
from agent_committee.prereg import load_prereg


def main() -> None:
    prereg = load_prereg("prereg.yaml")
    rows = power_table(prereg.min_gap, prereg.alpha, prereg.power)
    manifest = json.loads(Path("data/split_manifest.json").read_text(encoding="utf-8"))
    n_test = manifest.get("n_test")
    print(
        f"min_gap={prereg.min_gap} alpha={prereg.alpha} target power={prereg.power} "
        "(exact two-sided McNemar; min_gap is ASSUMED)"
    )
    print(f"{'discordance':>12} {'required N':>11} {'power@N_test':>13}")
    for r in rows:
        p = mcnemar_power(n_test, prereg.min_gap, r.discordance, prereg.alpha) if n_test else 0.0
        mark = "  <- headline" if r.discordance == HEADLINE_DISCORDANCE else ""
        print(f"{r.discordance:>12.2f} {r.required_n:>11,} {p:>13.3f}{mark}")
    print(f"N_test = {n_test:,}")


if __name__ == "__main__":
    main()
