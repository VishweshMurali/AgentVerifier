# Agent Verifier

Is your AI agent worth what it costs? Agent Verifier races an AI agent against simpler, cheaper
models on the same real messages and tells you, in plain numbers, which one to use and where the
answer would flip.

- **Deterministic code does all scoring and deciding.** LLMs only act as candidate models; they
  never grade or decide.
- **Fair, paired and pre-registered.** Every model answers the same messages. The rules, costs
  and thresholds are written down in `prereg.yaml` before the test is unsealed.
- **Honest verdicts.** `DEPLOY_AGENT`, `USE_SIMPLER`, `NEED_MORE_DATA` or `INVALID`, each with its
  evidence level, the checks that ran and an audit bundle. When it is too close to call, it says so.

Banking77 (77 kinds of bank support message) is the first task and proves the method with one
sealed test. CLINC150, Emotion and MASSIVE-en are ready for quick tests, and you can add your own
labelled messages.

## What you can run

| | What it is | Status |
|---|---|---|
| **Sealed test** | one fixed-size, pre-registered run on a test set nobody has seen (`sealed/`, hash committed) | Banking77 v1 done; never re-run |
| **Quick test** | an exploratory run on a sample of DEV messages, sized from the difference you care about, with early stopping | the live page; results are exploratory, not sealed |
| **Live page** | a scroll story: describe your agent, pick the kinds of message it handles, watch the race, see one mistake, set your costs, get the answer | local server |

In quick tests the agent is, by default, a **local Ollama model** (llama3.1:8b on your computer)
standing in for yours. You can plug in **your own agent** over HTTP instead
(`POST {"text", "labels"} -> {"label"}`), with your own price per answer.

## Quick start

Needs Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 24, and [Ollama](https://ollama.com)
with `llama3.1:8b` (not needed when you test your own agent).

```sh
uv sync
npm --prefix frontend ci
npm --prefix frontend run build:server
uv run --project server python -m verifier_server      # http://127.0.0.1:8000
```

A quick test from the command line:

```sh
uv run python -m agent_committee.quick --scenario banking77 --gap 0.10 --scope card_arrival,lost_or_stolen_card --yes
```

Check everything (what CI runs): `uv run python scripts/check.py`.

## Layout

```
src/agent_committee/   the pipeline: data, arms, stats, checks, decision engine, quick tests, runner
server/                FastAPI local server for the live page (own uv project)
frontend/              React + Vite + TypeScript page
mcp_server/            MCP data endpoints (own uv project)
scenarios/             per-dataset settings and preregs (CLINC150, Emotion, MASSIVE-en)
data/                  DEV splits and hashes (the sealed test lives in sealed/, never committed)
tests/                 pipeline tests
```

`CLAUDE.md` holds the locked design decisions and every command.

## What is not verified

- Docker images are written but not built or started.
- Quick tests are exploratory; only the sealed run is a verified verdict.
- The CLINC150, Emotion and MASSIVE-en preregs copy Banking77's assumed prices and thresholds.

## Data

DEV splits are derived from public datasets on Hugging Face: Banking77 (PolyAI), CLINC150 and
MASSIVE (Amazon), included here; Emotion (dair-ai/emotion) is not included, because its card
describes research and educational use. Its settings are included; build its split on your own
machine with `uv run python -m agent_committee.data --scenario emotion`. Check each dataset's own card for its
licence and terms before reusing the data.

The code is under the MIT licence (`LICENSE`); the datasets keep their own licences.
