"""Check the checks on DEV only (never TEST).

    uv run python scripts/validate_checks.py [--reps 100]

Tune, eval and pool come from one stratified split, so no drift check may fire between them, and
on random halves of held-out DEV the drift checks must fire at most `max_false_alarm_rate` of the
time. Score drift compares held-out scores only (tune vs eval); the layer's in-sample confidence
gap (pool vs eval) is printed as an overfitting signal. Also reports linear's tune gap, the Null
rate and how much of eval near-copies the pool, a preview of the leakage rule on DEV.
"""

import argparse

from agent_committee import checks
from agent_committee.arms.linear import make_layer
from agent_committee.arms.splits import make_arms_split
from agent_committee.data import load_dev
from agent_committee.prereg import load_prereg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args(argv)
    prereg = load_prereg()
    cfg, seed = prereg.checks, prereg.stats.seed
    split = make_arms_split(load_dev("data"))
    layer = make_layer(split.pool)

    def run(cases):
        decisions = [layer.decide(c.text) for c in cases]
        right = sum(d.choice == c.label for d, c in zip(decisions, cases, strict=True))
        return decisions, right

    pool_d, _ = run(split.pool)
    tune_d, tune_right = run(split.tune)
    eval_d, eval_right = run(split.eval)
    leak, kept = checks.leakage(split.eval, split.pool, cfg)
    held = [*split.tune, *split.eval]
    held_scores = [d.score for d in [*tune_d, *eval_d]]
    results = [
        checks.tune_gap("linear", tune_right, len(split.tune), eval_right, len(split.eval), cfg),
        checks.null_rate(sum(d.null for d in eval_d), len(split.eval), cfg),
        leak,
        checks.score_drift([d.score for d in tune_d], [d.score for d in eval_d], cfg, True),
        checks.domain_auc(
            [c.text for c in split.pool], [c.text for c in split.eval], cfg, True, seed
        ),
        checks.label_mix([c.label for c in split.pool], [c.label for c in split.eval], cfg),
        checks.false_alarm_rate([c.text for c in held], held_scores, cfg, args.reps, seed),
    ]
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'} {r.effect.value:<14} {r.name}: {r.detail}")
    in_sample = checks.score_drift([d.score for d in pool_d], [d.score for d in eval_d], cfg, True)
    print(f"overfitting signal: linear in-sample vs held-out confidence, {in_sample.detail}")
    copied = {c.id for c in split.eval} - {c.id for c in kept}
    for c in split.eval:
        if c.id in copied:
            print(f"  eval copy of pool: {c.text!r} => {c.label}")
    return 0 if all(r.passed for r in results if r.name != "leakage") else 1


if __name__ == "__main__":
    raise SystemExit(main())
