# Version 2 build plan: phases, workers, time (2026-11-06)

Plan only. Nothing here is approved. Version 1 stays frozen at `arms-frozen`. Scope comes from
`docs/NEXT_VERSION.md` (narrow version, prerequisites, "Datasets as plug-ins", customer-facing
clarity) and the 2026-11-05 review backlog. Every phase below is a question for the user.

## Phases
Time is working days for one worker. "Elapsed" assumes at most 2 workers at once (Sonnet), picked so
their files do not overlap. Waits for user decisions, Ollama run time and the design partner are NOT
included.

| # | Phase | Workers (files) | Days | Notes |
|---|---|---|---|---|
| P1a | Prerequisites, stats side: cost binds DEPLOY_AGENT; USE_SIMPLER mirror case; `find_copies` Jaccard; A/A test that can detect a bad rule; power of the full rule | Analysis (`analysis`, `decision`, `checks`, `power`) | 2 to 3 | Fake data only. Runs in parallel with P1b. |
| P1b | Prerequisites, model side: `llm.py` SQLite lock fixes (WAL, busy timeout, rollback, one transaction), response checks, served-digest check, usage kept on refusal | LLM layer (`llm.py`, throughput) | 2 to 3 | Fake transports only. |
| P1c | Prerequisites, arms: keep usage when context is refused, reject truncated or no-tool-turn tool calls, validate the 4th call, malformed-call rule (needs F2 decision) | Arms (`arms/`) | 1 to 2 | After P1b (uses its new error types). Alone. |
| P2 | Dataset plug-in: scenario interface, generalise `task: Literal["banking77"]`, loader, label set, scorer by name, own prereg, own sealed split, hash and seal guard | Pipeline (`schemas`, `prereg`, `scorers`, `runner`) plus `data.py`, `seal.py` (ownership: see decisions) | 2 to 3 | Banking77 becomes scenario 1; its v1 result must reproduce unchanged. |
| P3 | Comparator ladder and cascade: majority floor, linear, one more classic rung, cascade (cheap rung on confident cases, agent only on flagged) | Arms (`arms/`) | 2 to 3 | DEV runs use Ollama: serial, hours. |
| P4 | Data handling for real teams: grouped splits, minimum per label, enough-cases check | Analysis (`checks`, `power`) | 1 to 2 | Parallel with P3 (different files). |
| P5 | Agent adapter and trust boundary: HTTP or LangGraph adapter, consent, pinned version, usage reporting, dry-run, no-retention workspace | LLM layer (adapter) plus Server (`server/`) | 2 to 3 | Parallel pair. |
| P6 | Deployment-review report: plain one-page summary, "How we know" detail layer, glossary | Pipeline (`report`, templates) plus Frontend | 2 to 3 | Parallel pair. |
| P7 | v2 sealed run on a scenario: new prereg, freeze tag, one run | Hub runs it; prereg and tag are the user's | 1 plus Ollama hours | Serial. v1's run took hours and crashed once on the lock (fixed in P1b). |
| P8 | Design partner run on their data | the user | not estimated | Outside our control. |

Total build, P1 to P7: 15 to 23 worker-days, so about 8 to 12 working days elapsed with 2 workers,
plus decision waits and Ollama time. Reports to the user at the end of every phase.

## How the estimate was made
- v1 calibration (`git log`): Phase A on 10-31, arms frozen 11-04, sealed verdict 11-05, with two
  workers and stops for user decisions: about 5 to 6 days for four phases.
- Two sources disagree. "About a week" (the `docs-v2` branch) is the narrow version without the
  review backlog or the plug-in. "3 to 4 weeks" (earlier memory note) was the broad plan with a
  large ladder, intake chat, synthetic data and quick screen; that is now deferred. This plan uses
  the narrow version, plus the backlog (P1) and the plug-in (P2), which add roughly half again.

## Progress (2026-11-09)
P1 done (pushed). P2 done (9bcc731), P4 done (b6fc4b6), local. P3 approved and next. Owner decisions
taken: F2 valid only; per-scenario freeze tags; P3 arms (majority floor, Naive Bayes, cascade,
non-LLM rungs $0); main owns data.py and seal.py.

## Order and parallel pairs
P1a with P1b, then P1c; P2; P3 with P4; P5 and P6 as two pairs; P7. Ollama is single-process, so any
phase with real model runs (P3 DEV eval, P7) is serial whatever the worker count.

## Decisions only the user can make
1. Approve v2, and each phase before it starts (phases P1 and P2 edit files frozen at `arms-frozen`).
2. v2 prereg and freeze tag: user-owned; the sealed set for each scenario.
3. Plug-in ownership: `data.py` and `seal.py` are Hub-only and the Hub writes no feature code.
   Suggested: the Pipeline worker takes `data.py`, `seal.py` and their tests for v2, and the Hub
   reviews every `seal.py` diff before commit.
4. F2: count only valid tool use toward the quality floor (add a second count)?
5. Third rung (Naive Bayes or nearest neighbour), and how to price non-LLM rungs ($0 local, or a
   stated $ per 1,000 predictions).
6. The design partner and their task.
7. Who is editing `docs/NEXT_VERSION.md` (the "Datasets as plug-ins" section is uncommitted and was
   not written by the Hub).
8. Push now, or ask ChatGPT to re-check first? 5 commits are unpushed; the 3 worker changes
   (front-end proposals, R4-S1, R4-S2) are done but not yet reviewed or committed.
