# agent-committee

Decides, for a given scenario, whether an AI agent is worth deploying, statistically and cost-wise,
compared with simpler alternatives under fair, controlled, pre-registered conditions, and reports
where the economic decision flips. Nothing is deployed, and business impact is out of scope.
Banking77 is the first task and proves the method.

## Locked decisions (do not change without asking the user)
- LLMs only build/run candidate arms. **Deterministic code does ALL scoring and deciding.** Never use an LLM for scoring or decisions.
- Product: general verifier. Agents are plugged in; the system supplies baselines. First task: Banking77 intent classification (Hugging Face). ETL is out of scope.
- Shared first step: word + character logistic regression, trained on the pool, threshold set on tune, then frozen. It returns Choice, Score and Null.
- Arms (all use DEV data only):
  1. Linear model alone, no LLM. It always answers with its Choice.
  2. Single prompt with the top-5 shortlist and few-shot examples from DEV.
  3. Hand-built LangGraph reference agent with the shortlist (tools: retrieve similar DEV examples, read label definitions; then decide). No label list is given.
  v2 additions (approved 2026-11-09, P3): a majority-label floor; Naive Bayes as the third classic rung; a cascade
  (linear on confident cases, the agent only on Null-flagged ones). Non-LLM rungs are priced at $0 (local).
- Null is a flag for low-confidence cases (the lowest 10% on tune), not a score. All arms answer them, and accuracy on those cases is reported per arm.
- Agent rules: a no-tool answer is sent back once, the last call must answer, a malformed tool call scores wrong. Maximum 4 model calls per case.
- No RAG for now. `search` stays a single swappable tool.
- Test set is **SEALED**: stored separately (`sealed/`, gitignored) with a SHA-256 hash committed in `data/test.sha256`.
  No arm-building code may read it. Unsealing requires the scenario's freeze tag: `arms-frozen` for Banking77 (v1), `arms-frozen-<name>` for every
  other scenario (approved 2026-11-09). Banking77's v1 tag and test are used up; a v2 sealed run needs a new scenario.
- `prereg.yaml` is committed before unsealing: metric=accuracy, alpha=0.05, minimum meaningful gap, cost ranges (assumed), 3 price tables (assumed), decision policy.
- Stats: fixed sample; paired comparisons on the same cases; exact McNemar (k=1) and paired cluster bootstrap over cases; A/A simulation target ~5% false winners.
- Decision: E[C] = C_inference + P(error)*C_error, under each price table and across the error-cost range.
  Report break-even error cost with bootstrap CI **only if** the CI of the accuracy difference is entirely above 0.
  Otherwise the simpler arm dominates or verdict = NEED_MORE_DATA.
- Verdicts: `DEPLOY_AGENT | USE_SIMPLER | NEED_MORE_DATA | INVALID`. Latency is reported but excluded from the decision.
- Every verdict has an evidence level: `REAL`, `SYNTHETIC_CHECKED` or `SYNTHETIC`. `SYNTHETIC` alone never picks a winner: it gives `NEED_MORE_DATA` with the estimate shown, or `INVALID` when the run itself is broken (no scorer, failed floor, failed INVALID check).
- Each scenario declares a plain-code scorer, otherwise `INVALID`. The agent must pass a quality floor before the sealed run, otherwise `INVALID`.
- The headline is the break-even error cost. No error cost is invented. Each verdict saves an audit bundle.
- Stack: Python 3.12, uv, openai SDK -> Ollama (`http://localhost:11434/v1`, overridable with env `OLLAMA_BASE_URL`, model `llama3.1:8b`, temperature 0), LangGraph, scikit-learn, numpy, scipy, pydantic, datasets, SQLite, pytest, ruff, jinja2 (HTML report), pyyaml (approved by the user for prereg.yaml). Front end: React + Vite + TypeScript (approved 2026-11-04), built to the locked design in `design/frontend/SPEC.md`. Local server: FastAPI + uvicorn (approved 2026-11-04), in `server/` with its own pyproject. MCP data endpoints: fastmcp + Docker (approved 2026-11-09), in `mcp_server/` with its own pyproject.

## Phases (implement only the current one; STOP and report after each; wait for "continue")
- A: CLAUDE.md, pydantic schemas, prereg.yaml template, data loader (stratified DEV/TEST split + seed), sealed test + hash + unseal guard, throughput benchmark.
- B: power analysis script (required N at 80% power for the prereg minimum gap), LLM wrapper (SQLite cache, retries, timeout, token + latency logging).
- C (reopened for the shared linear layer and new arms): the 3 arms, evaluated on DEV only. Print the exact `git tag arms-frozen` command for the user; **do not tag yourself**.
- D: unseal guard check -> paired test run -> leakage check -> analysis (McNemar, cluster bootstrap, A/A) -> decision engine -> `results/verdict.json` -> `results/report.html`.

## Hard rules
- NEVER read the sealed test file outside the Phase D runner. Only `runner.py` may import `agent_committee.seal`; `data.py` may only *write* the sealed split at creation. A test (`tests/test_seal.py`) enforces this by scanning imports.
- NEVER add `Co-Authored-By` or any Claude/AI attribution to commits or PRs (the user's rule overrides defaults).
- Tests are written alongside each module (`tests/test_<module>.py`).
- Stop and ask before: adding dependencies not listed above, deleting files, changing locked decisions.
- No workarounds that weaken the product. On a design problem, stop and ask the user, even if the fix seems clear.
- Discuss design with the user before building. The user approves tasks; workers record them in `COORDINATION.md`.
- If a test still fails after 3 fix attempts: STOP and report.
- If Ollama is unreachable: STOP. Never mock silently. (Unit tests may use a fake transport; runtime never falls back to fakes.)

## Layout
```
src/agent_committee/
  schemas.py     pydantic models: Case, ArmPrediction, PriceTable, Prereg, Verdict, VerdictJSON
  prereg.py      load/validate prereg.yaml
  scenarios.py   scenario registry (Banking77 = scenario 1, v1 paths and tag); frozen with the arms
  catalog.py     dataset menu: Banking77, CLINC150, Emotion presets (pinned HF revisions) + your own file
  data.py        scenario loaders (Banking77, Hugging Face, local jsonl/csv), stratified split, writes DEV + sealed TEST + hashes
  seal.py        ONLY reader of sealed TEST; enforces the scenario's freeze tag + hash (Phase D runner only)
  throughput.py  calls/hour benchmark logic
  power.py       exact McNemar power and required N
  llm.py         LLM wrapper: SQLite cache, retries, phase/arm tags, usage logging
  scorers.py     plain-code scorers named by prereg.yaml (exact_label)
  checks.py      pre-registered overfitting, leakage, drift and bias checks; effects bind the verdict
  analysis.py    exact McNemar, paired case bootstrap, A/A on the full winner rule
  decision.py    decision engine: verdict rules, break-even, VerdictJSON
  runner.py      Phase D runner, the ONLY sealed-test reader; `--rehearse` runs it on DEV instead
  report.py      results/report.html from verdict.json (templates/report.html.j2)
  arms/          base protocol, config, splits, labels, linear (shared layer + arm 1), prompt, agent, tools, evaluate,
                 classic (majority + Naive Bayes rungs), cascade (derived from linear + agent; report-only)
server/          FastAPI local API + page (own uv project; never imports seal or runner)
mcp_server/      FastMCP data endpoints (own uv project; never imports seal or runner; Docker in mcp_server/)
scripts/benchmark_throughput.py   thin CLI around throughput.py
scripts/power_analysis.py         required N across discordance rates
scripts/validate_checks.py        checks the checks on DEV only (false alarms, leakage preview)
scripts/check.py                  one-command check of everything (CI: .github/workflows/ci.yml)
prereg.yaml      pre-registration (all costs/prices are ASSUMED)
data/            dev.jsonl, test.sha256, split_manifest.json (committed)
sealed/          test.jsonl (gitignored)
tests/
```

## Commands
- Check everything (what CI runs): `uv run python scripts/check.py` (`--no-install`, `--fast`)
- Tests: `uv run pytest`   Lint: `uv run ruff check . && uv run ruff format --check .`
- Build split (once): `uv run python -m agent_committee.data`
- DEV run of another scenario (non-LLM arms only until a per-scenario task phrase exists): `uv run python -m agent_committee.arms.evaluate --data-dir data/<name> --out scenarios/<name>/dev_report.json --arms majority,linear,naive_bayes`
- Benchmark: `uv run python scripts/benchmark_throughput.py`
- Rehearse Phase D on DEV (never opens TEST): `uv run python -m agent_committee.runner --rehearse`
- The one sealed run (after the user's `arms-frozen` tag): `uv run python -m agent_committee.runner`
- Dataset menu: `uv run python -m agent_committee.catalog` (`--choose NAME`)
- MCP data endpoints (tools: list_datasets, choose_dataset, list_scenarios, check_dataset, prereg_template, dev_stats, add_dataset): stdio `uv run --project mcp_server python -m verifier_mcp`; HTTP `... --http` (127.0.0.1:8765, read-only unless `--allow-writes`); Docker `docker compose -f mcp_server/compose.yaml up --build`; tests `uv run --project mcp_server pytest mcp_server/tests`
- Docker (page + MCP, Ollama on the host): `docker compose up --build`, page http://127.0.0.1:8000, MCP http://127.0.0.1:8765/mcp. Written, NOT yet built or tested. The sealed run stays a host command.
- Local server + live page: `npm --prefix frontend run build:server` then `uv run --project server python -m verifier_server` (http://127.0.0.1:8000); tests `uv run --project server pytest server/tests`
