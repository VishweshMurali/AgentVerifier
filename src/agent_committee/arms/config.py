"""Shared constants for the arms. Values measured by the C1 probe (see COORDINATION.md)."""

from dataclasses import dataclass

TEMPERATURE = 0.0
MODEL = "llama3.1:8b"
MODEL_DIGEST = "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"

# Ollama silently truncates the head of any prompt over this many tokens and still reports a
# smaller prompt_tokens, so truncation cannot be detected afterwards. Reject before sending.
CONTEXT_CAP_TOKENS = 4096
MAX_PROMPT_CHARS = 14_000  # about 3,250 tokens at 4.3 chars/token; sum all messages per call

K_SHOTS = 2  # few-shot examples per label in the single-prompt arm
RETRIEVAL_K = 5  # neighbours returned by the agent's retrieve tool
AGENT_MAX_TOOL_TURNS = 3  # at most 4 model calls per case

SHORTLIST_K = 5  # labels the shared linear layer passes to the prompt and agent arms
# 10th percentile of the layer's top score on the 488 tune cases, computed once. A flag only.
NULL_THRESHOLD = 0.486288


# What the LLM arms are told the task is (per scenario; approved 2026-11-10). The Banking77
# default rebuilds the v1 prompts and tool schema byte for byte, so its cache keys are unchanged.
TASK_KEYS = ("item", "items", "label_kind", "hint")


@dataclass(frozen=True)
class TaskPhrase:
    item: str  # "a banking customer query": what one case is
    items: str  # "banking queries": the labelled examples the search tool returns
    label_kind: str  # "intent": what a label is
    hint: str | None = None  # an optional extra sentence, e.g. what an "oos" label means

    def sentence(self) -> str:
        """The opening of both LLM arms' instructions (identical, so the comparison is fair)."""
        out = f"You classify {self.item} into exactly one {self.label_kind} label. "
        return out + (f"{self.hint.strip()} " if self.hint else "")

    def as_dict(self) -> dict[str, str]:
        return {
            k: v
            for k, v in zip(
                TASK_KEYS, (self.item, self.items, self.label_kind, self.hint), strict=True
            )
            if v is not None
        }


BANKING77_TASK = TaskPhrase("a banking customer query", "banking queries", "intent")


def task_from_dict(data: dict) -> TaskPhrase:
    """A TaskPhrase from JSON/YAML: exactly the known keys, all non-empty strings."""
    unknown = sorted(set(data) - set(TASK_KEYS))
    required = ("item", "items", "label_kind")
    if unknown or any(not isinstance(data.get(k), str) or not data[k].strip() for k in required):
        raise ValueError(
            f"task phrase needs non-empty {list(required)} (and optional hint): {data}"
        )
    if "hint" in data and (not isinstance(data["hint"], str) or not data["hint"].strip()):
        raise ValueError("task phrase hint must be a non-empty string")
    return TaskPhrase(data["item"], data["items"], data["label_kind"], data.get("hint"))
