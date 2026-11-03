"""Pydantic schemas shared by every module. Pure data + validation, no I/O."""

import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Verdict(StrEnum):
    DEPLOY_AGENT = "DEPLOY_AGENT"
    USE_SIMPLER = "USE_SIMPLER"
    NEED_MORE_DATA = "NEED_MORE_DATA"
    INVALID = "INVALID"


class ArmName(StrEnum):
    LINEAR = "linear"
    PROMPT = "prompt"
    AGENT = "agent"


class EvidenceLevel(StrEnum):
    REAL = "REAL"
    SYNTHETIC_CHECKED = "SYNTHETIC_CHECKED"
    SYNTHETIC = "SYNTHETIC"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Case(_Frozen):
    """One labelled example. `label` is the Banking77 intent name."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)


class ArmPrediction(_Frozen):
    """One arm's answer on one case. `pred=None` means no answer (scored wrong).

    No answer is unparseable output or a malformed tool call.
    `own_tool_calls` (total tool calls the model made), `sent_back` and `malformed` feed the
    agent quality floor; other arms keep the defaults.
    """

    case_id: str
    arm: ArmName
    pred: str | None
    prompt_tokens: int = Field(ge=0, default=0)
    completion_tokens: int = Field(ge=0, default=0)
    latency_s: float = Field(ge=0, default=0.0)
    own_tool_calls: int = Field(ge=0, default=0)
    sent_back: bool = False
    malformed: bool = False


class PriceTable(_Frozen):
    """USD per million tokens. Prices are assumptions, not quotes."""

    name: str = Field(min_length=1)
    input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)
    assumed: bool = True


class ErrorCostRange(_Frozen):
    """USD cost of one wrong prediction, swept between `low` and `high`."""

    low: float = Field(gt=0)
    high: float = Field(gt=0)
    assumed: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.low >= self.high:
            raise ValueError("error_cost.low must be < error_cost.high")
        return self


class SplitConfig(_Frozen):
    seed: int
    test_fraction: float = Field(gt=0, lt=1)


class StatsConfig(_Frozen):
    ci_level: float = Field(gt=0, lt=1, default=0.95)
    bootstrap_replicates: int = Field(ge=100, default=10_000)
    aa_simulations: int = Field(ge=100, default=1_000)
    seed: int = 0


class DecisionPolicy(_Frozen):
    """Machine-readable decision rules. Phase D implements exactly these, nothing else."""

    primary_comparison: Literal["agent_vs_best_simpler_on_dev"] = "agent_vs_best_simpler_on_dev"
    # Break-even error cost is reported only if the accuracy-difference CI is entirely above 0.
    breakeven_requires_ci_above_zero: Literal[True] = True
    # CI entirely above 0 AND point estimate >= min_gap  -> agent can win; else simpler/more data.
    require_point_estimate_at_least_min_gap: bool = True
    # CI includes 0 but the ci_level interval is too wide to rule out min_gap -> NEED_MORE_DATA.
    inconclusive_verdict: Literal["NEED_MORE_DATA"] = "NEED_MORE_DATA"
    latency_in_decision: Literal[False] = False


class LinearLayerConfig(_Frozen):
    """The shared layer's frozen settings. `null_threshold` flags cases; it scores none."""

    null_threshold: float = Field(ge=0, le=1)
    shortlist_k: int = Field(ge=1)


class ScorerSpec(_Frozen):
    """The scenario's plain-code scorer. `target` is its dotted path; no LLM may sit behind it."""

    name: str = Field(min_length=1)
    target: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


class QualityFloor(_Frozen):
    """What the agent must reach before the sealed run, otherwise the verdict is INVALID.

    It checks that the agent works as an agent. Accuracy is left to the comparison, so an agent
    that loses gets USE_SIMPLER, not INVALID.
    """

    min_own_tool_rate: float = Field(ge=0, le=1)
    max_send_back_rate: float = Field(ge=0, le=1)


class Prereg(_Frozen):
    """Contents of prereg.yaml. Committed before the sealed test set is unsealed.

    `linear`, `scorer` and `quality_floor` may stay unset while drafting; the pre-tag check and
    the unseal guard both refuse a prereg that leaves any of them out.
    """

    task: Literal["banking77"] = "banking77"
    metric: Literal["accuracy"] = "accuracy"
    alpha: float = Field(gt=0, lt=1)
    min_gap: float = Field(gt=0, lt=1, description="Smallest accuracy difference worth acting on")
    power: float = Field(gt=0, lt=1, default=0.8)
    split: SplitConfig
    stats: StatsConfig = StatsConfig()
    error_cost: ErrorCostRange
    price_tables: list[PriceTable]
    decision_policy: DecisionPolicy = DecisionPolicy()
    linear: LinearLayerConfig | None = None
    scorer: ScorerSpec | None = None
    quality_floor: QualityFloor | None = None

    @model_validator(mode="after")
    def _three_unique_tables(self) -> Self:
        if len(self.price_tables) != 3:
            raise ValueError("exactly 3 price tables are required")
        if len({t.name for t in self.price_tables}) != 3:
            raise ValueError("price table names must be unique")
        return self


class BreakEven(_Frozen):
    """Error cost (USD) above which the agent's lower error rate pays for its extra cost."""

    price_table: str
    value: float
    ci_low: float
    ci_high: float


class AuditBundle(_Frozen):
    """What a reader needs to retrace one verdict. Saved with every verdict.

    `predictions_sha256` stays None only on an INVALID verdict reached before the paired run.
    """

    git_commit: str = Field(min_length=1)
    prereg_sha256: str = Field(min_length=1)
    dev_report_sha256: str = Field(min_length=1)
    test_sha256: str = Field(min_length=1)
    model_digest: str = Field(min_length=1)
    layer_fingerprint: str = Field(min_length=1)
    predictions_sha256: str | None = None


class VerdictJSON(_Frozen):
    """Contents of results/verdict.json, written by the deterministic decision engine."""

    verdict: Verdict
    evidence_level: EvidenceLevel
    audit: AuditBundle
    reasons: list[str]
    scorer: str | None = None  # None means no declared scorer, which is INVALID
    quality_floor_passed: bool | None = None  # None means not checked, which is INVALID
    accuracy: dict[str, float]
    flagged_rate: float | None = None
    shortlist_coverage: float | None = None
    flagged_accuracy: dict[str, float | None] = {}  # per arm; None if no case was flagged
    accuracy_diff: float | None = None
    accuracy_diff_ci: tuple[float, float] | None = None
    mcnemar_p: float | None = None
    aa_false_winner_rate: float | None = None
    break_even: list[BreakEven] = []
    latency_s: dict[str, float] = {}  # reported only; never used in the decision

    @model_validator(mode="after")
    def _finite_and_ordered(self) -> Self:
        """No NaN/inf anywhere numeric, and every CI must satisfy lo <= hi."""
        numbers: list[float] = [*self.accuracy.values(), *self.latency_s.values()]
        for x in (
            self.accuracy_diff,
            self.mcnemar_p,
            self.aa_false_winner_rate,
            self.flagged_rate,
            self.shortlist_coverage,
            *self.flagged_accuracy.values(),
        ):
            if x is not None:
                numbers.append(x)
        if self.accuracy_diff_ci is not None:
            numbers.extend(self.accuracy_diff_ci)
        for b in self.break_even:
            numbers.extend((b.value, b.ci_low, b.ci_high))
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("verdict numbers must be finite (no NaN/inf)")
        if (
            self.accuracy_diff_ci is not None
            and self.accuracy_diff_ci[0] > self.accuracy_diff_ci[1]
        ):
            raise ValueError("accuracy_diff_ci must satisfy lo <= hi")
        if any(b.ci_low > b.ci_high for b in self.break_even):
            raise ValueError("break_even CI must satisfy ci_low <= ci_high")
        return self

    @model_validator(mode="after")
    def _break_even_guard(self) -> Self:
        """Break-even may only be reported when the accuracy-difference CI is entirely above 0."""
        if self.break_even and (self.accuracy_diff_ci is None or self.accuracy_diff_ci[0] <= 0):
            raise ValueError("break_even requires accuracy_diff_ci lower bound > 0")
        return self

    @model_validator(mode="after")
    def _deploy_needs_evidence(self) -> Self:
        """DEPLOY_AGENT is only valid with an accuracy-difference CI entirely above 0."""
        if self.verdict == Verdict.DEPLOY_AGENT and (
            self.accuracy_diff_ci is None or self.accuracy_diff_ci[0] <= 0
        ):
            raise ValueError("DEPLOY_AGENT requires accuracy_diff_ci lower bound > 0")
        return self

    @model_validator(mode="after")
    def _valid_verdicts_need_their_preconditions(self) -> Self:
        """Any verdict but INVALID needs a declared scorer, a passed floor and saved predictions."""
        if self.verdict == Verdict.INVALID:
            return self
        if self.scorer is None:
            raise ValueError("a scenario without a declared scorer must be INVALID")
        if self.quality_floor_passed is not True:
            raise ValueError("an agent that did not pass the quality floor must be INVALID")
        if self.audit.predictions_sha256 is None:
            raise ValueError("audit.predictions_sha256 is required unless the verdict is INVALID")
        return self

    @model_validator(mode="after")
    def _synthetic_alone_decides_nothing(self) -> Self:
        """SYNTHETIC evidence shows an estimate and never picks a winner."""
        if self.evidence_level == EvidenceLevel.SYNTHETIC:
            if self.verdict in (Verdict.DEPLOY_AGENT, Verdict.USE_SIMPLER):
                raise ValueError("SYNTHETIC evidence gives only NEED_MORE_DATA")
            if self.verdict == Verdict.NEED_MORE_DATA and not self.accuracy:
                raise ValueError("NEED_MORE_DATA on SYNTHETIC evidence must show the estimate")
        return self
