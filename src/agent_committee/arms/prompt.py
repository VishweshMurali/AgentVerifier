"""Arm 2: one prompt built from the linear layer's shortlist.

The prompt names the top candidate labels with K fixed examples each. Candidate examples are static
per label, not retrieved per query, which keeps this arm distinct from the agent.
"""

from agent_committee.arms.config import K_SHOTS, MODEL_DIGEST
from agent_committee.arms.labels import LabelBook
from agent_committee.arms.linear import LinearLayer
from agent_committee.arms.parsing import PromptTooLong, guarded_chat, parse_answer
from agent_committee.llm import LLMClient
from agent_committee.schemas import ArmName, ArmPrediction, Case

ANSWER_MAX_TOKENS = 32
INSTRUCTIONS = (
    "You classify a banking customer query into exactly one intent label. "
    "Reply with one label from the candidates below, spelled exactly as shown, and nothing else. "
    "Never invent a new label.\n"
    "Candidates with examples (query => label):\n"
)


def build_system_prompt(book: LabelBook, shortlist: tuple[str, ...], k: int = K_SHOTS) -> str:
    lines = [f"{text} => {label}" for label in shortlist for text in book.get(label).examples[:k]]
    return INSTRUCTIONS + "\n".join(lines)


class PromptArm:
    name = ArmName.PROMPT

    def __init__(self, client: LLMClient, book: LabelBook, layer: LinearLayer) -> None:
        self._client = client
        self._book = book
        self._layer = layer

    def fit(self, pool: list[Case]) -> None:
        """Nothing to fit: the label book and the layer were both built from the pool."""

    def predict(self, case_id: str, text: str) -> ArmPrediction:
        shortlist = self._layer.decide(text).shortlist
        messages = [
            {"role": "system", "content": build_system_prompt(self._book, shortlist)},
            {"role": "user", "content": text},
        ]
        try:
            result = guarded_chat(
                self._client, messages, arm=self.name.value, max_tokens=ANSWER_MAX_TOKENS
            )
        except PromptTooLong:
            # never sent, so the model cannot answer it: scored wrong and counted, not a crash
            return ArmPrediction(case_id=case_id, arm=self.name, pred=None, too_long=True)
        return ArmPrediction(
            case_id=case_id,
            arm=self.name,
            pred=parse_answer(result, self._book),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_s=result.latency_s,
        )


def make_arm(
    pool: list[Case], book: LabelBook, layer: LinearLayer, client: LLMClient | None = None
) -> PromptArm:
    client = client or LLMClient(phase="build", model_digest=MODEL_DIGEST)
    arm = PromptArm(client, book, layer)
    arm.fit(pool)
    return arm
