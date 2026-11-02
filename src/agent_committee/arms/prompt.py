"""Arm 2: one prompt with static few-shot examples from the pool.

The same K medoid examples per label go into every prompt. They are not retrieved per case, which
keeps this arm distinct from the agent.
"""

from agent_committee.arms.config import K_SHOTS, MODEL_DIGEST
from agent_committee.arms.labels import LabelBook
from agent_committee.arms.parsing import guarded_chat, parse_answer
from agent_committee.llm import LLMClient
from agent_committee.schemas import ArmName, ArmPrediction, Case

ANSWER_MAX_TOKENS = 32
INSTRUCTIONS = (
    "You classify banking customer queries into exactly one intent label. "
    "Reply with one label from the list below, spelled exactly as shown, and nothing else. "
    "Never invent a new label.\n"
)


def build_system_prompt(book: LabelBook, k: int = K_SHOTS) -> str:
    lines = [f"{text} => {info.label}" for info in book.infos for text in info.examples[:k]]
    return (
        INSTRUCTIONS
        + "Examples (query => label):\n"
        + "\n".join(lines)
        + "\nValid labels:\n"
        + "\n".join(book.labels)
    )


class PromptArm:
    name = ArmName.PROMPT

    def __init__(self, client: LLMClient, book: LabelBook) -> None:
        self._client = client
        self._book = book
        self._system: str | None = None

    def fit(self, pool: list[Case]) -> None:
        """The prompt comes from the label book, which was itself derived from the pool."""
        self._system = build_system_prompt(self._book)

    def predict(self, case_id: str, text: str) -> ArmPrediction:
        if self._system is None:
            raise RuntimeError("fit() first")
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": text},
        ]
        result = guarded_chat(
            self._client, messages, arm=self.name.value, max_tokens=ANSWER_MAX_TOKENS
        )
        return ArmPrediction(
            case_id=case_id,
            arm=self.name,
            pred=parse_answer(result, self._book),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_s=result.latency_s,
        )


def make_arm(pool: list[Case], book: LabelBook, client: LLMClient | None = None) -> PromptArm:
    client = client or LLMClient(phase="build", model_digest=MODEL_DIGEST)
    arm = PromptArm(client, book)
    arm.fit(pool)
    return arm
