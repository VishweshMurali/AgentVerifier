"""Renders results/report.html from a validated VerdictJSON. Presents the verdict, never makes it.

The decision engine has already chosen the verdict; this module only formats what it saved.
`expected_cost` is the prereg formula E[C] = C_inference + P(error) * C_error, used for display.
"""

import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from agent_committee.schemas import CheckEffect, CheckResult, VerdictJSON

TEMPLATE = "report.html.j2"
GATING = (CheckEffect.INVALID, CheckEffect.NEED_MORE_DATA)


def expected_cost(inference_usd: float, accuracy: float, error_cost: float) -> float:
    """Expected USD cost per case: inference plus error rate times the cost of one error."""
    return inference_usd + (1 - accuracy) * error_cost


def _num(x: float | None, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _usd(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.2f}" if abs(x) >= 1 else f"${x:.6f}"


def _price_tables(verdict: VerdictJSON) -> list[str]:
    """Price table names in first-seen order across arms and break-even rows."""
    names: list[str] = []
    for arm in verdict.arms:
        names.extend(n for n in arm.inference_cost if n not in names)
    names.extend(b.price_table for b in verdict.break_even if b.price_table not in names)
    return names


def _gating_failure(check: CheckResult) -> bool:
    return not check.passed and check.effect in GATING


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(Path(__file__).with_name("templates")),
        autoescape=select_autoescape(["html", "j2"], default_for_string=True, default=True),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["num"] = _num
    env.filters["usd"] = _usd
    return env


def render_report(verdict: VerdictJSON) -> str:
    """The report page as one self-contained HTML string."""
    tables = _price_tables(verdict)
    low, high = verdict.error_cost_range or (None, None)
    arm_rows = []
    for arm in verdict.arms:
        costs = [arm.inference_cost.get(t) for t in tables]
        expected = [
            (
                None if c is None or low is None else expected_cost(c, arm.accuracy, low),
                None if c is None or high is None else expected_cost(c, arm.accuracy, high),
            )
            for c in costs
        ]
        role = (
            "agent"
            if arm.name == verdict.agent_arm
            else "simpler"
            if arm.name == verdict.simpler_arm
            else ""
        )
        arm_rows.append({"arm": arm, "role": role, "costs": costs, "expected": expected})
    checks = [{"check": c, "gating_failure": _gating_failure(c)} for c in verdict.checks]
    return (
        _environment()
        .get_template(TEMPLATE)
        .render(
            v=verdict,
            verdict=verdict.verdict.value,
            evidence=verdict.evidence_level.value,
            tables=tables,
            error_low=low,
            error_high=high,
            arm_rows=arm_rows,
            checks=checks,
            audit=verdict.audit.model_dump(),
        )
    )


def write_report(verdict: VerdictJSON, path: str | Path) -> Path:
    """Write the report as UTF-8 and return its path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(verdict), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> Path:
    """CLI: python -m agent_committee.report results/verdict.json [--out results/report.html]."""
    parser = argparse.ArgumentParser(description="Render the verdict report as HTML.")
    parser.add_argument("verdict_json", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/report.html"))
    args = parser.parse_args(argv)
    verdict = VerdictJSON.model_validate_json(args.verdict_json.read_text(encoding="utf-8"))
    out = write_report(verdict, args.out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
