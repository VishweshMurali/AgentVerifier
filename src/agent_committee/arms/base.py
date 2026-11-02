"""The contract every arm implements."""

from typing import Protocol

from agent_committee.schemas import ArmName, ArmPrediction, Case


class Arm(Protocol):
    """A candidate classifier.

    `fit` receives the labelled pool. `predict` receives only the case id and text: gold labels
    never reach it. The caller builds the LLMClient, so an arm never chooses the phase it logs to.
    """

    name: ArmName

    def fit(self, pool: list[Case]) -> None: ...

    def predict(self, case_id: str, text: str) -> ArmPrediction: ...
