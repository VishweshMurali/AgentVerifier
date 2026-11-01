import pytest
from pydantic import ValidationError

from agent_committee.schemas import (
    ArmName,
    ArmPrediction,
    Case,
    ErrorCostRange,
    Prereg,
    Verdict,
    VerdictJSON,
)


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
        ArmPrediction(case_id="1", arm="tfidf", pred="a", prompt_tokens=-1)


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
    common = {"verdict": Verdict.DEPLOY_AGENT, "reasons": ["r"], "accuracy": {"agent": 0.9}}
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
_BASE = {"verdict": Verdict.USE_SIMPLER, "reasons": ["r"], "accuracy": {"agent": 0.9}}


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
