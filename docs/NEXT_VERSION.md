# Next version: ideas and limits (noted 2026-11-05)

Not approved for building. Each item changes a locked decision, and the arms stay frozen until the
Banking77 sealed run is done. Discuss with the user before starting any of them.

## Is it novel?
The method is not: baselines vs agents ("AI Agents That Matter", Kapoor et al. 2024), McNemar,
bootstrap, pre-registration, break-even cost and eval tools (LangSmith, Braintrust, promptfoo) all
exist. What is less common is the package: rules locked before the test, binding integrity checks,
a go/no-go verdict that can refuse (NEED_MORE_DATA, INVALID), the break-even error cost as the
headline, and baselines supplied for you. No market search done yet.

## Where it works
Tasks with a plain-code scorer and a few thousand labelled cases: classification and routing,
exact extraction (dates, amounts, IDs), text-to-SQL (execute and compare), code (tests decide),
structured output (schema checks).

## Where it fails or misleads
1. Open-ended output (summaries, chat, writing): no plain-code scorer, so INVALID by design.
2. Small data: a 2-point gap needs about 3,000 cases.
3. Unequal error costs: every mistake is priced the same.
4. Agents that take actions: side effects and partial success are not measured.
5. Rare but important cases: overall accuracy hides them.
6. Production differs from the test: one sealed test, one point in time.
7. Cost covers model calls only: not build/maintenance effort, latency or user experience.
8. Narrow conclusion: it is about this agent on this model, not agents in general.

## Improvements, most valuable first
1. Cascade arm: Linear answers confident cases, the agent only the flagged ones (the Null flag
   exists already).
2. Cost per type of mistake (a cost matrix the user sets).
3. Fewer cases needed: sequential testing that stops early when the answer is clear.
4. More scorer types: execution-based (SQL, code tests) and schema checks.
5. Plug-in interface: any agent connects through an adapter (HTTP or LangGraph).
6. Per-segment results: by intent or difficulty, not one overall number.
7. Monitoring: rerun the checks on live traffic (e.g. LangSmith) and alert when the decision flips.

## Also planned
- One-command check of everything, an end-to-end test, and GitHub CI (approved 2026-11-05,
  waiting until the user finishes reviewing).
