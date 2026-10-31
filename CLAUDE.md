# agent-committee

Decides whether an AI agent is worth deploying by comparing it with simpler alternatives under
fair, controlled, pre-registered conditions, and reports where the economic decision flips.

## Locked decisions (do not change without asking the user)
- LLMs only build/run candidate arms. **Deterministic code does ALL scoring and deciding.** Never use an LLM for scoring or decisions.
- MVP task: Banking77 intent classification (Hugging Face). ETL is out of scope.
- Arms (all use DEV data only):
  1. TF-IDF + logistic regression
  2. Single prompt with few-shot examples from DEV
  3. Hand-built LangGraph reference agent (tools: retrieve similar DEV examples, read label definitions; then decide)
- Test set is **SEALED**: stored separately (`sealed/`, gitignored) with a SHA-256 hash committed in `data/test.sha256`.
  No arm-building code may read it. Unsealing requires the git tag `arms-frozen` to exist.
- `prereg.yaml` is committed before unsealing: metric=accuracy, alpha=0.05, minimum meaningful gap, cost ranges (assumed), 3 price tables (assumed), decision policy.
- Stats: fixed sample; paired comparisons on the same cases; exact McNemar (k=1) and paired cluster bootstrap over cases; A/A simulation target ~5% false winners.
- Decision: E[C] = C_inference + P(error)*C_error, under each price table and across the error-cost range.
  Report break-even error cost with bootstrap CI **only if** the CI of the accuracy difference is entirely above 0.
  Otherwise the simpler arm dominates or verdict = NEED_MORE_DATA.
- Verdicts: `DEPLOY_AGENT | USE_SIMPLER | NEED_MORE_DATA | INVALID`. Latency is reported but excluded from the decision.
- Stack: Python 3.12, uv, openai SDK -> Ollama (`http://localhost:11434/v1`, model `qwen2.5:7b`, temperature 0), LangGraph, scikit-learn, numpy, scipy, pydantic, datasets, SQLite, pytest, ruff, jinja2 (HTML report), pyyaml (approved by the user for prereg.yaml).

## Phases (implement only the current one; STOP and report after each; wait for "continue")
- A: CLAUDE.md, pydantic schemas, prereg.yaml template, data loader (stratified DEV/TEST split + seed), sealed test + hash + unseal guard, throughput benchmark.
- B: power analysis script (required N at 80% power for the prereg minimum gap), LLM wrapper (SQLite cache, retries, timeout, token + latency logging).
- C: the 3 arms, evaluated on DEV only. Print the exact `git tag arms-frozen` command for the user; **do not tag yourself**.
- D: unseal guard check -> paired test run -> leakage check -> analysis (McNemar, cluster bootstrap, A/A) -> decision engine -> `results/verdict.json` -> `results/report.html`.

## Hard rules
- NEVER read the sealed test file outside the Phase D runner. Only `runner.py` may import `agent_committee.seal`; `data.py` may only *write* the sealed split at creation. A test (`tests/test_seal.py`) enforces this by scanning imports.
- NEVER add `Co-Authored-By` or any Claude/AI attribution to commits or PRs (the user's rule overrides defaults).
- Tests are written alongside each module (`tests/test_<module>.py`).
- Stop and ask before: adding dependencies not listed above, deleting files, changing locked decisions.
- If a test still fails after 3 fix attempts: STOP and report.
- If Ollama is unreachable: STOP. Never mock silently. (Unit tests may use a fake transport; runtime never falls back to fakes.)

## Layout
```
src/agent_committee/
  schemas.py     pydantic models: Case, ArmPrediction, PriceTable, Prereg, Verdict, VerdictJSON
  prereg.py      load/validate prereg.yaml
  data.py        Banking77 load, deterministic stratified split, writes DEV + sealed TEST + hashes
  seal.py        ONLY reader of sealed TEST; enforces arms-frozen tag + hash (Phase D runner only)
  throughput.py  calls/hour benchmark logic
scripts/benchmark_throughput.py   thin CLI around throughput.py
prereg.yaml      pre-registration (all costs/prices are ASSUMED)
data/            dev.jsonl, test.sha256, split_manifest.json (committed)
sealed/          test.jsonl (gitignored)
tests/
```

## Commands
- Tests: `uv run pytest`   Lint: `uv run ruff check . && uv run ruff format --check .`
- Build split (once): `uv run python -m agent_committee.data`
- Benchmark: `uv run python scripts/benchmark_throughput.py`
