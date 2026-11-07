"""The decision engine: turns paired results on the sealed cases into a verdict. Plain code, no LLM.

Order of rules, each from prereg.yaml or CLAUDE.md:
1. INVALID: no declared scorer, a failed quality floor, or any failed INVALID check.
2. NEED_MORE_DATA: any failed NEED_MORE_DATA check, or SYNTHETIC evidence alone.
3. DEPLOY_AGENT: exact McNemar p < alpha, the accuracy-difference CI entirely above 0, the
   point estimate at least min_gap, and the agent's expected cost E[C] lower in at least one
   pre-registered scenario (price table x end of the error-cost range). The break-even error cost
   per price table is the headline.
4. USE_SIMPLER: the CI's upper bound is below min_gap (no gain worth acting on is plausible) and
   the simpler arm is either no more expensive per case under any price table, or has the lower
   E[C] in every scenario.
5. Otherwise NEED_MORE_DATA: the data still allow a gain of min_gap or more.
Latency is carried through for the report and never used here.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from agent_committee.analysis import (
    aa_false_winner_rate,
    bootstrap_means,
    mcnemar_exact,
    percentile_ci,
)
from agent_committee.schemas import (
    ArmResult,
    AuditBundle,
    BreakEven,
    CheckEffect,
    CheckResult,
    EvidenceLevel,
    Prereg,
    PriceTable,
    Verdict,
    VerdictJSON,
)

# bootstrap replicates inside each A/A simulation; the main CI uses prereg's full count
AA_BOOTSTRAP_REPLICATES = 200


@dataclass(frozen=True)
class ArmRun:
    """One arm's per-case outcomes on the paired cases, aligned by position with every other arm."""

    name: str
    correct: np.ndarray
    prompt_tokens: np.ndarray
    completion_tokens: np.ndarray
    latency_s: np.ndarray


def inference_cost(prompt_tokens: float, completion_tokens: float, price: PriceTable) -> float:
    """USD per case for the given mean token counts."""
    return (
        prompt_tokens * price.input_usd_per_mtok + completion_tokens * price.output_usd_per_mtok
    ) / 1e6


def break_even_cost(
    agent_cost: float, agent_error: float, simpler_cost: float, simpler_error: float
) -> float:
    """Error cost at which E[C] of the two arms is equal; 0 if the agent is never more expensive.

    E[C] = C_inference + P(error) * C_error. Infinite when the agent is not more accurate.
    """
    if agent_cost <= simpler_cost and agent_error <= simpler_error:
        return 0.0
    if agent_error >= simpler_error:
        return math.inf
    return max(0.0, (agent_cost - simpler_cost) / (simpler_error - agent_error))


def _arm_result(arm: ArmRun, flagged: np.ndarray, prereg: Prereg) -> ArmResult:
    n = len(arm.correct)
    prompt, completion = float(np.mean(arm.prompt_tokens)), float(np.mean(arm.completion_tokens))
    return ArmResult(
        name=arm.name,
        n=n,
        correct=int(np.sum(arm.correct)),
        accuracy=float(np.mean(arm.correct)),
        prompt_tokens_mean=prompt,
        completion_tokens_mean=completion,
        latency_s_mean=float(np.mean(arm.latency_s)),
        flagged_accuracy=float(np.mean(arm.correct[flagged])) if flagged.any() else None,
        inference_cost={t.name: inference_cost(prompt, completion, t) for t in prereg.price_tables},
    )


def _break_evens(
    agent: ArmRun, simpler: ArmRun, prereg: Prereg
) -> tuple[list[BreakEven], list[str]]:
    """Break-even per price table with a bootstrap CI from the same case resamples as the diff."""
    cols = np.column_stack(
        [
            agent.correct.astype(float),
            simpler.correct.astype(float),
            agent.prompt_tokens,
            agent.completion_tokens,
            simpler.prompt_tokens,
            simpler.completion_tokens,
        ]
    )
    boots = bootstrap_means(cols, prereg.stats.bootstrap_replicates, prereg.stats.seed)
    point = cols.mean(axis=0)
    out, notes = [], []
    for table in prereg.price_tables:

        def be(row: np.ndarray, table: PriceTable = table) -> float:
            return break_even_cost(
                inference_cost(row[2], row[3], table),
                1 - row[0],
                inference_cost(row[4], row[5], table),
                1 - row[1],
            )

        value = be(point)
        lo, hi = percentile_ci(np.array([be(r) for r in boots]), prereg.stats.ci_level)
        if not all(math.isfinite(x) for x in (value, lo, hi)):
            notes.append(f"Break-even under {table.name} has an unbounded CI; not reported.")
            continue
        out.append(BreakEven(price_table=table.name, value=value, ci_low=lo, ci_high=hi))
    return out, notes


def _scenarios(
    agent: dict[str, float], simpler: dict[str, float], diff: float, prereg: Prereg
) -> tuple[list[str], list[str], list[str]]:
    """Scenarios (price table x end of the error-cost range) won by the agent, the simpler arm, or
    neither (equal E[C]). E[C] = C_inference + P(error) * C_error, and P(error) is lower for the
    agent by `diff`. E[C] is linear in the error cost, so the range ends decide every cost between.
    """
    agent_wins, simpler_wins, ties = [], [], []
    for table in agent:
        for cost in (prereg.error_cost.low, prereg.error_cost.high):
            gap = (agent[table] - simpler[table]) - diff * cost  # agent E[C] minus simpler E[C]
            label = f"{table}@${cost:g}"
            (agent_wins if gap < 0 else simpler_wins if gap > 0 else ties).append(label)
    return agent_wins, simpler_wins, ties


def _aa_rate(
    agent: ArmRun,
    simpler: ArmRun,
    prereg: Prereg,
    audit: AuditBundle,
    scorer: str | None,
    run_aa: bool,
) -> float | None:
    """A/A false-winner rate of the whole winner rule: each simulation calls `decide` itself on
    outcomes swapped between the arms (token counts stay with their arm) and counts a DEPLOY_AGENT
    in either direction."""
    st = prereg.stats
    if not run_aa or scorer is None:
        return None
    quick = prereg.model_copy(
        update={"stats": st.model_copy(update={"bootstrap_replicates": AA_BOOTSTRAP_REPLICATES})}
    )
    counter = iter(range(10**9))

    def names_winner(x: np.ndarray, y: np.ndarray, swapped: bool) -> bool:
        first, second = (simpler, agent) if swapped else (agent, simpler)
        stats = quick.stats.model_copy(update={"seed": st.seed + next(counter)})
        out = decide(
            replace(first, correct=x),
            replace(second, correct=y),
            [],
            np.zeros(len(x), dtype=bool),
            quick.model_copy(update={"stats": stats}),
            [],
            scorer,
            True,
            EvidenceLevel.REAL,
            audit,
            0,
            run_aa=False,
        )
        return out.verdict == Verdict.DEPLOY_AGENT

    return aa_false_winner_rate(
        agent.correct, simpler.correct, st.aa_simulations, names_winner, st.seed
    )


def _no_break_even_cause(
    p: float, lo: float, diff: float, prereg: Prereg, agent_cheaper: bool = True
) -> str:
    """The first unmet condition for reporting a break-even, in the order they are tested."""
    if lo <= 0:
        return "the accuracy difference CI is not entirely above 0"
    if p >= prereg.alpha:
        return f"McNemar p={p:.3g} is not below alpha={prereg.alpha}"
    if diff < prereg.min_gap:
        return f"the gain {diff:+.3f} is below the {prereg.min_gap} minimum gap"
    if not agent_cheaper:
        return "the agent has no lower expected cost in any price table and error cost"
    return "a pre-registered check or rule stopped the decision before it"


def decide(
    agent: ArmRun,
    simpler: ArmRun,
    others: list[ArmRun],
    flagged: np.ndarray,
    prereg: Prereg,
    checks: list[CheckResult],
    scorer: str | None,
    floor_passed: bool | None,
    evidence: EvidenceLevel,
    audit: AuditBundle,
    n_dropped: int,
    shortlist_coverage: float | None = None,
    run_aa: bool = True,
) -> VerdictJSON:
    """Apply the pre-registered rules to the paired outcomes and return the verdict.

    `run_aa=False` skips the A/A simulation (which itself calls this function on swapped data).
    """
    flagged = np.asarray(flagged, dtype=bool)
    arms = [agent, simpler, *others]
    n_cases = len(agent.correct)
    for each in arms:
        for arr in (each.correct, each.prompt_tokens, each.completion_tokens, each.latency_s):
            if len(arr) != n_cases:
                raise ValueError(f"arm {each.name!r} does not have the paired length {n_cases}")
    if len(flagged) != n_cases:
        raise ValueError(f"flagged has length {len(flagged)}, expected {n_cases}")
    results = [_arm_result(a, flagged, prereg) for a in arms]
    st = prereg.stats
    only_a, only_s, p = mcnemar_exact(agent.correct, simpler.correct)
    diff_cols = np.column_stack([agent.correct.astype(float) - simpler.correct.astype(float)])
    boots = bootstrap_means(diff_cols, st.bootstrap_replicates, st.seed)[:, 0]
    diff = float(np.mean(diff_cols))
    lo, hi = percentile_ci(boots, st.ci_level)
    aa = _aa_rate(agent, simpler, prereg, audit, scorer, run_aa)

    failed = [c for c in checks if not c.passed]
    reasons: list[str] = []
    break_even: list[BreakEven] = []
    agent_costs = results[0].inference_cost
    simpler_costs = results[1].inference_cost
    simpler_not_dearer = all(simpler_costs[t] <= agent_costs[t] for t in agent_costs)
    agent_wins, simpler_wins, ties = _scenarios(agent_costs, simpler_costs, diff, prereg)
    simpler_lowest_everywhere = not agent_wins and not ties

    if scorer is None:
        verdict = Verdict.INVALID
        reasons.append("No plain-code scorer was declared.")
    elif floor_passed is not True:
        verdict = Verdict.INVALID
        reasons.append("The agent did not pass the quality floor.")
    elif any(c.effect == CheckEffect.INVALID for c in failed):
        verdict = Verdict.INVALID
        reasons += [f"Check failed: {c.name} ({c.detail})." for c in failed]
    elif any(c.effect == CheckEffect.NEED_MORE_DATA for c in failed):
        verdict = Verdict.NEED_MORE_DATA
        reasons += [
            f"Check failed: {c.name} ({c.detail})."
            for c in failed
            if c.effect == CheckEffect.NEED_MORE_DATA
        ]
    elif evidence == EvidenceLevel.SYNTHETIC:
        verdict = Verdict.NEED_MORE_DATA
        reasons.append("Synthetic evidence alone never picks a winner; the estimate is shown.")
    elif p < prereg.alpha and lo > 0 and diff >= prereg.min_gap and agent_wins:
        verdict = Verdict.DEPLOY_AGENT
        reasons.append(
            f"The agent is more accurate by {diff:+.3f} (CI {lo:+.3f} to {hi:+.3f}, "
            f"McNemar p={p:.3g}), at least the {prereg.min_gap} minimum gap."
        )
        break_even, notes = _break_evens(agent, simpler, prereg)
        reasons += notes
    elif hi < prereg.min_gap and (simpler_not_dearer or simpler_lowest_everywhere):
        verdict = Verdict.USE_SIMPLER
        why = (
            "costs no more per case under every price table"
            if simpler_not_dearer
            else "has the lower expected cost in every price table across the error-cost range"
        )
        reasons.append(
            f"A gain of {prereg.min_gap} or more is ruled out (CI {lo:+.3f} to {hi:+.3f}), and "
            f"{simpler.name} {why}."
        )
    else:
        verdict = Verdict.NEED_MORE_DATA
        reasons.append(
            f"The CI ({lo:+.3f} to {hi:+.3f}) still allows a gain of {prereg.min_gap} or more."
            if hi >= prereg.min_gap
            else f"{simpler.name} costs more per case under some price table; no clear winner."
        )
    reasons.append(
        f"Lower expected cost per scenario (price table@error cost): {agent.name}: "
        f"{', '.join(agent_wins) or 'none'}; {simpler.name}: {', '.join(simpler_wins) or 'none'}"
        + (f"; equal: {', '.join(ties)}." if ties else ".")
    )
    if verdict != Verdict.DEPLOY_AGENT:
        cause = _no_break_even_cause(p, lo, diff, prereg, agent_cheaper=bool(agent_wins))
        reasons.append(f"No break-even error cost: {cause}.")

    return VerdictJSON(
        verdict=verdict,
        evidence_level=evidence,
        audit=audit,
        reasons=reasons,
        scorer=scorer,
        quality_floor_passed=floor_passed,
        accuracy={r.name: r.accuracy for r in results},
        flagged_rate=float(np.mean(flagged)),
        shortlist_coverage=shortlist_coverage,
        flagged_accuracy={r.name: r.flagged_accuracy for r in results},
        accuracy_diff=diff,
        accuracy_diff_ci=(lo, hi),
        mcnemar_p=p,
        aa_false_winner_rate=aa,
        break_even=break_even,
        latency_s={r.name: r.latency_s_mean for r in results},
        checks=checks,
        agent_arm=agent.name,
        simpler_arm=simpler.name,
        n_cases=n_cases,
        n_dropped=n_dropped,
        mcnemar_counts=(only_a, only_s),
        arms=results,
        error_cost_range=(prereg.error_cost.low, prereg.error_cost.high),
        alpha=prereg.alpha,
        ci_level=st.ci_level,
        min_gap=prereg.min_gap,
    )
