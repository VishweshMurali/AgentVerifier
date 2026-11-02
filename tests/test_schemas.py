import pytest
from pydantic import ValidationError

from agent_committee.schemas import (
    ArmName,
    ArmPrediction,
    AuditBundle,
    Case,
    ErrorCostRange,
    EvidenceLevel,
    LinearLayerConfig,
    Prereg,
    QualityFloor,
    ScorerSpec,
    Verdict,
    VerdictJSON,
)

_AUDIT = {
    "git_commit": "abc",
    "prereg_sha256": "p",
    "dev_report_sha256": "d",
    "test_sha256": "t",
    "model_digest": "m",
    "layer_fingerprint": "f",
    "predictions_sha256": "h",
}
_PRECONDITIONS = {
    "evidence_level": EvidenceLevel.REAL,
    "audit": _AUDIT,
    "scorer": "exact_label",
    "quality_floor_passed": True,
}


def _prereg_dict(**over):
    base = {
        "alpha": 0.05,
        "min_gap": 0.02,
        "split": {"seed": 1, "test_fraction": 0.25},
        "error_cost": {"low": 1, "high": 10},
        "price_tables": [
            {"name": n, "input_usd_per_mtok": 1, "output_usd_per_mtok": 2} for n in "abc"
        ],
    }
    base.update(over)
    return base


def test_verdict_has_exactly_the_four_locked_values():
    assert {v.value for v in Verdict} == {
        "DEPLOY_AGENT",
        "USE_SIMPLER",
        "NEED_MORE_DATA",
        "INVALID",
    }


def test_case_rejects_empty_fields_and_is_frozen():
    with pytest.raises(ValidationError):
        Case(id="", text="x", label="y")
    c = Case(id="1", text="x", label="y")
    with pytest.raises(ValidationError):
        c.text = "z"


def test_case_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Case(id="1", text="x", label="y", extra=1)


def test_prediction_allows_unparseable_none_and_rejects_negative_tokens():
    assert ArmPrediction(case_id="1", arm=ArmName.AGENT, pred=None).pred is None
    with pytest.raises(ValidationError):
        ArmPrediction(case_id="1", arm="linear", pred="a", prompt_tokens=-1)


def test_prereg_valid():
    p = Prereg.model_validate(_prereg_dict())
    assert p.metric == "accuracy" and len(p.price_tables) == 3


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_prereg_rejects_bad_alpha(alpha):
    with pytest.raises(ValidationError):
        Prereg.model_validate(_prereg_dict(alpha=alpha))


def test_prereg_rejects_non_accuracy_metric():
    with pytest.raises(ValidationError):
        Prereg.model_validate(_prereg_dict(metric="f1"))


def test_prereg_requires_exactly_three_unique_price_tables():
    tables = _prereg_dict()["price_tables"]
    with pytest.raises(ValidationError):
        Prereg.model_validate(_prereg_dict(price_tables=tables[:2]))
    dup = [dict(t, name="a") for t in tables]
    with pytest.raises(ValidationError):
        Prereg.model_validate(_prereg_dict(price_tables=dup))


def test_error_cost_range_must_be_ordered_and_positive():
    with pytest.raises(ValidationError):
        ErrorCostRange(low=10, high=1)
    with pytest.raises(ValidationError):
        ErrorCostRange(low=0, high=1)


def test_verdict_json_blocks_break_even_unless_ci_entirely_above_zero():
    be = {"price_table": "a", "value": 5.0, "ci_low": 3.0, "ci_high": 8.0}
    common = {
        **_PRECONDITIONS,
        "verdict": Verdict.DEPLOY_AGENT,
        "reasons": ["r"],
        "accuracy": {"agent": 0.9},
    }
    assert VerdictJSON(**common, accuracy_diff_ci=(0.01, 0.05), break_even=[be])
    with pytest.raises(ValidationError):
        VerdictJSON(**common, accuracy_diff_ci=(-0.01, 0.05), break_even=[be])
    with pytest.raises(ValidationError):
        VerdictJSON(**common, accuracy_diff_ci=(0.0, 0.05), break_even=[be])  # lower bound == 0
    with pytest.raises(ValidationError):
        VerdictJSON(**common, break_even=[be])  # no CI at all
    simpler = {**common, "verdict": Verdict.USE_SIMPLER}
    assert VerdictJSON(**simpler).break_even == []


_BE = {"price_table": "a", "value": 5.0, "ci_low": 3.0, "ci_high": 8.0}
_BASE = {
    **_PRECONDITIONS,
    "verdict": Verdict.USE_SIMPLER,
    "reasons": ["r"],
    "accuracy": {"agent": 0.9},
}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_verdict_json_rejects_non_finite_numbers(bad):
    with pytest.raises(ValidationError):
        VerdictJSON(**_BASE, accuracy_diff_ci=(bad, 0.05))
    with pytest.raises(ValidationError):
        VerdictJSON(**_BASE, accuracy_diff=bad)
    with pytest.raises(ValidationError):
        VerdictJSON(**_BASE, mcnemar_p=bad)
    with pytest.raises(ValidationError):
        VerdictJSON(**{**_BASE, "accuracy": {"agent": bad}})
    with pytest.raises(ValidationError):
        VerdictJSON(**_BASE, latency_s={"agent": bad})
    with pytest.raises(ValidationError):
        VerdictJSON(**_BASE, flagged_rate=bad)
    with pytest.raises(ValidationError):
        VerdictJSON(**_BASE, shortlist_coverage=bad)
    with pytest.raises(ValidationError):
        VerdictJSON(**_BASE, flagged_accuracy={"agent": bad})
    with pytest.raises(ValidationError):
        VerdictJSON(**_BASE, accuracy_diff_ci=(0.01, 0.05), break_even=[{**_BE, "value": bad}])


def test_verdict_json_rejects_inverted_ci():
    with pytest.raises(ValidationError):
        VerdictJSON(**_BASE, accuracy_diff_ci=(0.05, 0.01))
    with pytest.raises(ValidationError):
        VerdictJSON(
            **_BASE, accuracy_diff_ci=(0.01, 0.05), break_even=[{**_BE, "ci_low": 9, "ci_high": 1}]
        )


def test_deploy_agent_requires_ci_lower_bound_above_zero():
    deploy = {**_BASE, "verdict": Verdict.DEPLOY_AGENT}
    assert VerdictJSON(**deploy, accuracy_diff_ci=(0.01, 0.05)).verdict == Verdict.DEPLOY_AGENT
    for ci in (None, (0.0, 0.05), (-0.01, 0.05)):
        with pytest.raises(ValidationError):
            VerdictJSON(**deploy, accuracy_diff_ci=ci)
    # the other verdicts need no CI evidence
    for v in (Verdict.USE_SIMPLER, Verdict.NEED_MORE_DATA, Verdict.INVALID):
        assert VerdictJSON(**{**_BASE, "verdict": v}).verdict == v


def test_arm_name_has_linear():
    assert ArmName("linear") == ArmName.LINEAR


def test_prediction_carries_the_quality_floor_counts():
    p = ArmPrediction(case_id="1", arm=ArmName.AGENT, pred="a", own_tool_calls=2, sent_back=True)
    assert p.own_tool_calls == 2 and p.sent_back is True and p.malformed is False
    assert ArmPrediction(case_id="1", arm=ArmName.LINEAR, pred="a").own_tool_calls == 0
    with pytest.raises(ValidationError):
        ArmPrediction(case_id="1", arm=ArmName.AGENT, pred="a", own_tool_calls=-1)


def test_prereg_new_sections_are_optional_and_validated():
    p = Prereg.model_validate(_prereg_dict())
    assert p.linear is None and p.scorer is None and p.quality_floor is None
    full = Prereg.model_validate(
        _prereg_dict(
            linear={"null_threshold": 0.3, "shortlist_k": 5},
            scorer={"name": "exact_label", "target": "agent_committee.scorers:exact_label"},
            quality_floor={
                "min_own_tool_rate": 0.9,
                "max_send_back_rate": 0.1,
                "max_points_below_linear": 0.05,
            },
        )
    )
    assert full.linear == LinearLayerConfig(null_threshold=0.3, shortlist_k=5)
    assert full.quality_floor.min_own_tool_rate == 0.9


@pytest.mark.parametrize(
    ("make", "kwargs"),
    [
        (LinearLayerConfig, {"null_threshold": 1.5, "shortlist_k": 5}),
        (LinearLayerConfig, {"null_threshold": 0.5, "shortlist_k": 0}),
        (ScorerSpec, {"name": "", "target": "a.b:c"}),
        (ScorerSpec, {"name": "x", "target": "not a path"}),
        (
            QualityFloor,
            {"min_own_tool_rate": 1.1, "max_send_back_rate": 0.1, "max_points_below_linear": 0.05},
        ),
    ],
)
def test_new_prereg_sections_reject_bad_values(make, kwargs):
    with pytest.raises(ValidationError):
        make(**kwargs)


def test_every_verdict_needs_an_evidence_level_and_an_audit_bundle():
    for missing in ("evidence_level", "audit"):
        with pytest.raises(ValidationError):
            VerdictJSON(**{k: v for k, v in _BASE.items() if k != missing})


def test_audit_requires_real_values_and_predictions_may_be_absent():
    assert AuditBundle(**{**_AUDIT, "predictions_sha256": None}).predictions_sha256 is None
    with pytest.raises(ValidationError):
        AuditBundle(**{**_AUDIT, "git_commit": ""})


@pytest.mark.parametrize(
    "bad",
    [
        {"scorer": None},
        {"quality_floor_passed": None},
        {"quality_floor_passed": False},
        {"audit": {**_AUDIT, "predictions_sha256": None}},
    ],
)
@pytest.mark.parametrize(
    "verdict", [Verdict.USE_SIMPLER, Verdict.NEED_MORE_DATA, Verdict.DEPLOY_AGENT]
)
def test_missing_preconditions_force_invalid(bad, verdict):
    ci = {"accuracy_diff_ci": (0.01, 0.05)}
    with pytest.raises(ValidationError):
        VerdictJSON(**{**_BASE, "verdict": verdict, **bad}, **ci)
    assert VerdictJSON(**{**_BASE, "verdict": Verdict.INVALID, **bad}, **ci)


def test_synthetic_alone_can_only_be_need_more_data_or_invalid():
    synthetic = {**_BASE, "evidence_level": EvidenceLevel.SYNTHETIC}
    for v in (Verdict.DEPLOY_AGENT, Verdict.USE_SIMPLER):
        with pytest.raises(ValidationError):
            VerdictJSON(**{**synthetic, "verdict": v}, accuracy_diff_ci=(0.01, 0.05))
    assert VerdictJSON(**{**synthetic, "verdict": Verdict.NEED_MORE_DATA})
    assert VerdictJSON(**{**synthetic, "verdict": Verdict.INVALID})
    with pytest.raises(ValidationError):  # the estimate must be shown
        VerdictJSON(**{**synthetic, "verdict": Verdict.NEED_MORE_DATA, "accuracy": {}})


def test_synthetic_checked_can_reach_a_real_verdict():
    checked = {**_BASE, "evidence_level": EvidenceLevel.SYNTHETIC_CHECKED}
    assert VerdictJSON(**checked).verdict == Verdict.USE_SIMPLER
