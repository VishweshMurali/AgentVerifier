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

## Known defects in v1 (found 2026-11-05, frozen, not fixed)
Both are in files frozen under `arms-frozen`, so the Banking77 sealed run used them as they are.
1. Leakage check overstates overlap (`checks.find_copies`). The 5-gram vectorizer is fit on DEV
   only, so test 5-grams unseen in DEV are dropped and the union is too small. Example: "lost card
   the" vs "lost card the top" has true Jaccard 0.69 but is flagged at 0.90; 445 of 4,000 random
   pairs were misjudged, always towards flagging. Effect: some of the 37 cases dropped as copies
   may not be copies. This only removes cases, so it cannot make an arm look better. Fix: fit the
   vectorizer on both sides, or compute exact set Jaccard.
2. Decision rule picks the agent on accuracy alone (`decision.decide`, line 198). DEPLOY_AGENT can
   be given even when the agent costs more across the whole pre-registered error-cost range; cost
   only shows up as the break-even. No effect on Banking77, where the agent trails. Fix: require
   the agent's expected cost to be lower somewhere in the range, otherwise NEED_MORE_DATA or
   USE_SIMPLER.

## Improvements, most valuable first
1. Cascade arm: Linear answers confident cases, the agent only the flagged ones (the Null flag
   exists already).
2. Cost per type of mistake (a cost matrix the user sets).
3. Fewer cases needed: sequential testing that stops early when the answer is clear.
4. More scorer types: execution-based (SQL, code tests) and schema checks.
5. Plug-in interface: any agent connects through an adapter (HTTP or LangGraph).
6. Per-segment results: by intent or difficulty, not one overall number.
7. Monitoring: rerun the checks on live traffic (e.g. LangSmith) and alert when the decision flips.

## Datasets as plug-ins (user's request, 2026-11-06)
Goal: plug and play. Any labelled dataset becomes a scenario without touching core code; a set of
default datasets ships ready to run. Today Banking77 is hard-wired (`data.load_banking77()`,
`schemas.py` `task: Literal["banking77"]`), so this comes first in the harness work.

A scenario plug-in declares:
- a loader (Hugging Face name, or a local `cases.jsonl` / CSV with `text` and `label` columns);
- the label set (and optional label definitions for the agent's tool);
- a plain-code scorer by name (`exact_label` now; `execution_match`, `schema_valid` later);
- its own prereg file (metric, minimum gap, cost ranges, price tables) and split seed;
- its own sealed split, hash and seal guard, built the same way as Banking77's.

Bring your own data: `verify --data my_cases.jsonl` uses the same interface. Input checks run
first: required columns, label counts, duplicates and near-copies, and whether the number of
cases is enough for the minimum gap (from the power analysis).

Default datasets (details and fit checks in `docs/CANDIDATE_TASKS.md`):

| Dataset | Task | Scorer | Status |
|---|---|---|---|
| Banking77 | intent routing, 77 labels | exact label | v1 done; becomes default scenario 1 |
| CLINC150 | intents + out-of-scope | exact label | next; tests "I don't know" handling |
| Spider | text-to-SQL | execution match | needs the execution scorer |
| BIRD-SQL | text-to-SQL, messy schemas | execution match | after Spider |
| Bitext support | support tickets | exact label | demo only: machine-generated, SYNTHETIC evidence |
| CoNLL-2003 / Few-NERD, MAVEN | extraction | span F1 | later: needs a per-case right/wrong rule (locked decision) |

Rules that stay: deterministic scoring and deciding; a scenario without a plain-code scorer is
INVALID; each scenario's test set is sealed and its prereg committed before unsealing.

## Cost monitoring with LangSmith and Langfuse (user's idea, 2026-11-05)
Goal: every model call's tokens and cost are captured automatically and shown on a dashboard and
on the page, without the user setting anything up.
- One tracing interface in the LLM wrapper with switchable backends (LangSmith, Langfuse, off),
  picked by env variables. LangGraph traces natively to LangSmith; the openai SDK has drop-in
  wrappers for both.
- Ollama calls cost $0 to the tracers: load the prereg price tables as custom model prices, so
  dashboards and the verdict use the same numbers.
- The verdict never reads cost from a tracer (external, can change). Deterministic code keeps
  using our own token log; a new check compares our token counts with the tracer's.
- Privacy: traces carry the prompts. Sealed-run cases must not go to a cloud service, so either
  self-host Langfuse (Docker, open source) or turn tracing off or redact prompts in the sealed run.
- llm.py is frozen until the sealed run, so this waits for the next version. The front end already
  has a provider slot for it (frontend/src/data/index.ts).
- Suggested order: Langfuse first (self-hostable, keeps data local), LangSmith second.

## Product focus: outside review (2026-11-05)
The review's verdict: a strong idea as a focused tool, much weaker as "evaluate any agent".
- The buyer's question: is this agent worth paying for when something simpler might do? A
  break-even answer ("it pays off only if each avoided mistake is worth at least $3") is
  actionable, and so is a credible "keep the simpler system". The Banking77 run is that second case.
- First customer: a team already deciding whether to deploy an agent on a high-volume, measurable
  task (support routing, structured extraction) that has a candidate agent, historical examples
  and a rough cost per mistake.
- The hard part is enough representative, correctly labelled cases. More baselines do not fix
  that, and synthetic data makes demos easier without making the decision more reliable.
- What is ours: the decision workflow (a credible cheap comparator, a fair test, where the
  economics flip). Cost-aware evaluation itself has precedent ("AI Agents That Matter").
- Defer until the first use case proves out: the automatic recommender, a general intake chat and
  tracing integrations (the LangSmith and Langfuse section above).
- Validation that counts: a team changes a real deployment decision because of the report, then
  comes back to evaluate its next agent.

Our check against the project:
- Agrees with what we have: plain-code scorer, strong cheap baseline, break-even headline,
  verdicts that can refuse, and SYNTHETIC evidence alone never picking a winner.
- Labelled data is the binding constraint ("Where it fails" item 2: about 3,000 cases for a
  2-point gap). This raises "Fewer cases needed" (sequential testing) and makes help with
  labelling (spot checks, label-quality checks) a likely next need.
- Reorders the list: Cascade arm, cost per type of mistake and the plug-in interface serve the
  first customer directly; monitoring and tracing move later.
- Next step it implies: one real team's routing or extraction task, fresh labelled cases, and a
  report built for their deployment review.

## Also planned
- Done 2026-11-05: one-command check, end-to-end test, GitHub CI, and the static page on GitHub
  Pages (https://vishweshmurali.github.io/AgentVerifier/).
