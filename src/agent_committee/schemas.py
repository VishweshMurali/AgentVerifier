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
    TFIDF = "tfidf"
    PROMPT = "prompt"
    AGENT = "agent"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Case(_Frozen):
    """One labelled example. `label` is the Banking77 intent name."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)


class ArmPrediction(_Frozen):
    """One arm's answer on one case. `pred=None` means unparseable output (scored wrong)."""

    case_id: str
    arm: ArmName
    pred: str | None
    prompt_tokens: int = Field(ge=0, default=0)
    completion_tokens: int = Field(ge=0, default=0)
    latency_s: float = Field(ge=0, default=0.0)


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


class Prereg(_Frozen):
    """Contents of prereg.yaml. Committed before the sealed test set is unsealed."""

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


class VerdictJSON(_Frozen):
    """Contents of results/verdict.json, written by the deterministic decision engine."""

    verdict: Verdict
    reasons: list[str]
    accuracy: dict[str, float]
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
        for x in (self.accuracy_diff, self.mcnemar_p, self.aa_false_winner_rate):
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
