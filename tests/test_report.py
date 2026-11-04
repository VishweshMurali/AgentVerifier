"""Tests for the HTML verdict report. The report presents a verdict; it never decides one."""

from pathlib import Path

import pytest

from agent_committee.report import expected_cost, main, render_report, write_report
from agent_committee.schemas import (
    ArmResult,
    AuditBundle,
    BreakEven,
    CheckEffect,
    CheckResult,
    EvidenceLevel,
    Verdict,
    VerdictJSON,
)

COMMIT = "3412adc0ffee"
TABLES = {"cheap": 0.0001, "mid": 0.0004, "premium": 0.002}


def _audit(predictions: str | None = "f" * 64) -> AuditBundle:
    return AuditBundle(
        git_commit=COMMIT,
        prereg_sha256="a" * 64,
        dev_report_sha256="b" * 64,
        test_sha256="c" * 64,
        model_digest="llama3.1:8b@sha256:d" + "d" * 10,
        layer_fingerprint="e" * 16,
        predictions_sha256=predictions,
    )


def _arms() -> list[ArmResult]:
    return [
        ArmResult(
            name="linear",
            n=1000,
            correct=900,
            accuracy=0.9,
            prompt_tokens_mean=0,
            completion_tokens_mean=0,
            latency_s_mean=0.001,
            flagged_accuracy=0.5,
            inference_cost=dict.fromkeys(TABLES, 0.0),
        ),
        ArmResult(
            name="agent",
            n=1000,
            correct=940,
            accuracy=0.94,
            prompt_tokens_mean=2100.5,
            completion_tokens_mean=80.2,
            latency_s_mean=3.2,
            flagged_accuracy=None,
            inference_cost=TABLES,
        ),
    ]


def _common() -> dict:
    return {
        "audit": _audit(),
        "scorer": "exact_label",
        "quality_floor_passed": True,
        "accuracy": {"linear": 0.9, "agent": 0.94},
        "agent_arm": "agent",
        "simpler_arm": "linear",
        "n_cases": 1000,
        "n_dropped": 3,
        "arms": _arms(),
        "error_cost_range": (0.5, 50.0),
        "mcnemar_counts": (70, 30),
        "latency_s": {"linear": 0.001, "agent": 3.2},
    }


def use_simpler() -> VerdictJSON:
    return VerdictJSON(
        verdict=Verdict.USE_SIMPLER,
        evidence_level=EvidenceLevel.REAL,
        reasons=["CI includes 0 <script>alert(1)</script>"],
        accuracy_diff=0.01,
        accuracy_diff_ci=(-0.01, 0.03),
        mcnemar_p=0.2,
        aa_false_winner_rate=0.049,
        checks=[
            CheckResult(
                name="null_rate",
                value=0.2,
                low=0.05,
                high=0.15,
                passed=False,
                effect=CheckEffect.REPORT,
            )
        ],
        **_common(),
    )


def deploy_agent() -> VerdictJSON:
    return VerdictJSON(
        verdict=Verdict.DEPLOY_AGENT,
        evidence_level=EvidenceLevel.REAL,
        reasons=["accuracy difference CI entirely above 0"],
        accuracy_diff=0.04,
        accuracy_diff_ci=(0.02, 0.06),
        mcnemar_p=0.0001,
        aa_false_winner_rate=0.05,
        break_even=[
            BreakEven(price_table="cheap", value=0.0025, ci_low=0.0017, ci_high=0.005),
            BreakEven(price_table="premium", value=12.34, ci_low=8.5, ci_high=25.75),
        ],
        **_common(),
    )


def invalid() -> VerdictJSON:
    return VerdictJSON(
        verdict=Verdict.INVALID,
        evidence_level=EvidenceLevel.REAL,
        audit=_audit(None),
        reasons=["split drift found on our own sealed split"],
        accuracy={},
        checks=[
            CheckResult(
                name="score_ks_split",
                value=0.4,
                high=0.1,
                passed=False,
                effect=CheckEffect.INVALID,
                detail="KS D above limit",
            ),
            CheckResult(
                name="leak_rate", value=0.0, high=0.05, passed=True, effect=CheckEffect.REPORT
            ),
        ],
    )


def test_expected_cost_math() -> None:
    assert expected_cost(0.01, 0.9, 10.0) == pytest.approx(1.01)
    assert expected_cost(0.0, 1.0, 100.0) == 0.0
    assert expected_cost(0.5, 0.0, 2.0) == pytest.approx(2.5)


def test_use_simpler_says_no_break_even() -> None:
    html = render_report(use_simpler())
    assert "USE_SIMPLER" in html
    assert "linear" in html and "agent" in html
    assert "No break-even error cost is reported" in html
    assert "reported only, not used in the decision" in html
    assert COMMIT in html


def test_deploy_agent_shows_break_even_values() -> None:
    html = render_report(deploy_agent())
    assert "DEPLOY_AGENT" in html
    assert "No break-even error cost is reported" not in html
    for text in ("$0.002500", "$0.001700", "$0.005000", "$12.34", "$8.50", "$25.75"):
        assert text in html
    # expected cost for agent on premium at high error cost: 0.002 + 0.06 * 50 = 3.002
    assert "$3.00" in html
    assert "70" in html and "30" in html


def test_invalid_shows_failed_gating_check_and_missing_predictions() -> None:
    html = render_report(invalid())
    assert "INVALID" in html
    assert "score_ks_split" in html
    assert "FAILED, binds the verdict" in html
    assert html.count("FAILED, binds the verdict") == 1
    assert COMMIT in html
    assert "predictions_sha256" in html
    assert "n/a" in html


def test_report_only_failure_is_not_marked_as_binding() -> None:
    html = render_report(use_simpler())
    assert "null_rate" in html
    assert "FAILED, binds the verdict" not in html
    assert "no (report only)" in html


def test_html_is_escaped() -> None:
    html = render_report(use_simpler())
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_page_is_self_contained() -> None:
    html = render_report(deploy_agent())
    assert "<script" not in html
    assert "http://" not in html and "https://" not in html
    assert 'name="viewport"' in html


def test_write_report(tmp_path: Path) -> None:
    out = write_report(deploy_agent(), tmp_path / "results" / "report.html")
    assert out.exists()
    assert "DEPLOY_AGENT" in out.read_text(encoding="utf-8")


def test_cli_writes_report(tmp_path: Path) -> None:
    src = tmp_path / "verdict.json"
    src.write_text(invalid().model_dump_json(), encoding="utf-8")
    out = tmp_path / "report.html"
    assert main([str(src), "--out", str(out)]) == out
    assert "score_ks_split" in out.read_text(encoding="utf-8")
