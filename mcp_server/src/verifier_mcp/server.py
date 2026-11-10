"""MCP data endpoints for the Agent Verifier (FastMCP).

Read tools (always on): list_datasets (the menu), list_scenarios, check_dataset,
prereg_template, dev_stats.
Write tools (stdio, or HTTP started with --allow-writes): choose_dataset (a menu option; without
one it returns the menu, since the 2026-07-28 MCP protocol has no server-initiated pick-list) and
add_dataset (your own file). Each writes one scenario.yaml, which can be deleted again.

Rules this module keeps (tests enforce them):
- It never opens the sealed TEST set: no import of `agent_committee.seal` or `runner`, and no
  tool returns case text, only counts and hashes.
- Building a split is NOT an endpoint. It fixes a scenario's sealed TEST forever, the same kind
  of act as the freeze tag, so only the owner runs it, from the CLI. `check_dataset` prints the
  exact command once its checks pass.
- Local datasets must live in the repo's gitignored `sources/` folder.
- The prereg (costs, prices, minimum gap) stays the owner's: `prereg_template` returns a template
  with TODO markers and writes nothing.
"""

from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from yaml import YAMLError

from agent_committee import catalog, data
from agent_committee.scenarios import (
    SCENARIOS,
    HuggingFaceSource,
    LocalSource,
    ScenarioFile,
    TaskPhraseSpec,
    add_scenario,
    file_scenarios,
    get_scenario,
    read_scenario_file,
)

INSTRUCTIONS = (
    "Data endpoints of the Agent Verifier. Typical flow: list_datasets -> choose_dataset (or "
    "add_dataset for your own file) -> check_dataset -> the owner "
    "writes the scenario's prereg.yaml (prereg_template helps) -> the owner builds the split once "
    "from the CLI (check_dataset gives the command) -> dev_stats. No tool ever returns sealed "
    "TEST cases, and no tool can build a split."
)

PREREG_TEMPLATE = """# Pre-registration for scenario {name}. Fill every TODO yourself, then commit
# it BEFORE the freeze tag `arms-frozen-{name}`. Costs and prices are assumptions you own.
task: {name}
metric: accuracy
alpha: 0.05
min_gap: TODO  # smallest accuracy difference worth acting on, e.g. 0.02
split:
  seed: TODO  # any integer, fixed once
  test_fraction: 0.25
  grouped: false  # true if cases share a source (same customer or thread); needs a group column
  min_cases_per_label: 30
error_cost:
  low: TODO  # USD per wrong answer, low end
  high: TODO  # high end
price_tables: TODO  # exactly 3 tables: name, input_usd_per_mtok, output_usd_per_mtok
scorer:
  name: exact_label
  target: agent_committee.scorers:exact_label
# linear: filled from THIS dataset's DEV report after the DEV evaluation (null_threshold and
#   shortlist_k), never from Banking77's prereg; the split comes first, so leave it out until then
# quality_floor and checks: start from prereg.yaml (Banking77) and review every limit.
"""


def build_server(repo: str | Path = ".", allow_writes: bool = True) -> FastMCP:
    root = Path(repo).resolve()
    mcp = FastMCP("agent-verifier-data", instructions=INSTRUCTIONS)
    read_only = {"readOnlyHint": True, "openWorldHint": False}

    @mcp.tool(annotations=read_only)
    def list_datasets() -> list[dict[str, Any]]:
        """The dataset menu: standard datasets to pick from (Banking77, CLINC150, Emotion) and
        'your own dataset', each with its status and which arms can run on it."""
        return catalog.menu(root)

    @mcp.tool(annotations=read_only)
    def list_scenarios() -> list[dict[str, Any]]:
        """Every scenario: built-in (Banking77) and declared by file, with what exists for it."""
        out = []
        for name in [*SCENARIOS, *file_scenarios(root)]:
            try:
                s = get_scenario(name, root)
                source = read_scenario_file(name, root).source.kind if s.definition else "built-in"
            except (
                ValueError,
                OSError,
                YAMLError,
            ) as exc:  # one broken scenario.yaml must not hide the rest
                out.append({"name": name, "error": str(exc)})
                continue
            out.append(
                {
                    "name": name,
                    "source": source,
                    "has_prereg": (root / s.prereg_path).is_file(),
                    "split_built": (root / s.data_dir / data.DEV_FILE).is_file(),
                    "arms_ready": s.arms_ready,
                    "freeze_tag": s.freeze_tag,
                }
            )
        return out

    @mcp.tool(annotations={**read_only, "openWorldHint": True})
    def check_dataset(name: str) -> dict[str, Any]:
        """Dry run: load the dataset and run the input checks (unique ids, cases per label, groups,
        expected test size against the prereg's min_test_n). Writes nothing; counts only."""
        result = data.check_scenario(get_scenario(name, root), root)
        ready = not result["problems"] and result["prereg"] == "found" and not result["split_built"]
        # the owner runs this once, by hand: it fixes the scenario's sealed TEST forever
        result["build_command"] = (
            f"uv run python -m agent_committee.data --scenario {name}" if ready else None
        )
        return result

    @mcp.tool(annotations=read_only)
    def prereg_template(name: str) -> str:
        """A prereg.yaml template for the scenario, TODOs for the owner's values. Not written."""
        get_scenario(name, root)
        return PREREG_TEMPLATE.format(name=name)

    @mcp.tool(annotations=read_only)
    def dev_stats(name: str) -> dict[str, Any]:
        """Counts from the scenario's DEV split and its manifest. Never opens TEST."""
        return data.dev_stats(get_scenario(name, root), root)

    if allow_writes:

        @mcp.tool(annotations={"destructiveHint": False, "idempotentHint": True})
        def choose_dataset(option: str | None = None) -> Any:
            """Pick a standard dataset from the menu (see list_datasets). Without `option`, the
            menu is returned so the user can pick; call again with their choice. Downloads
            nothing and builds no split; choosing again only reports the status."""
            if option is None:
                return {"pick_one": catalog.menu(root), "how": "call again with option=<name>"}
            return catalog.choose(option, root)

        @mcp.tool(annotations={"destructiveHint": False, "idempotentHint": False})
        def add_dataset(
            name: str,
            kind: str,
            path: str | None = None,
            dataset: str | None = None,
            text: str | None = None,
            label: str | None = None,
            splits: list[str] | None = None,
            config: str | None = None,
            group: str | None = None,
            description: str = "",
            task_item: str | None = None,
            task_items: str | None = None,
            task_label_kind: str | None = None,
            task_hint: str | None = None,
        ) -> dict[str, Any]:
            """Declare a dataset as scenario `name` (writes scenarios/<name>/scenario.yaml once).

            kind="local": `path` to a .jsonl or .csv under sources/ with id, text, label (and
            optional group) columns. kind="huggingface": `dataset`, `text`, `label` column names,
            `splits` to pool, optional `config` and `group` column. Downloads nothing.

            Task phrase (optional; without it only the non-LLM arms can run, so no verdict): the
            LLM arms are told "You classify {task_item} into exactly one {task_label_kind}
            label.", the search tool returns labelled {task_items}, and `task_hint` is one
            optional extra sentence.
            """
            if name in catalog.PRESETS:
                raise ValueError(f"{name!r} is a menu option: use choose_dataset")
            if kind == "local":
                source = LocalSource(kind="local", path=path or "")
            elif kind == "huggingface":
                source = HuggingFaceSource(
                    kind="huggingface",
                    dataset=dataset or "",
                    text=text or "",
                    label=label or "",
                    splits=splits or [],
                    config=config,
                    group=group,
                )
            else:
                raise ValueError("kind must be 'local' or 'huggingface'")
            phrase_parts = (task_item, task_items, task_label_kind)
            if any(phrase_parts) and not all(phrase_parts):
                raise ValueError("a task phrase needs task_item, task_items and task_label_kind")
            phrase = (
                TaskPhraseSpec(
                    item=task_item, items=task_items, label_kind=task_label_kind, hint=task_hint
                )
                if all(phrase_parts)
                else None
            )
            spec = ScenarioFile(
                name=name, description=description, source=source, task_phrase=phrase
            )
            written = add_scenario(spec, root)
            return {"scenario": name, "written": written.relative_to(root).as_posix()}

    return mcp
