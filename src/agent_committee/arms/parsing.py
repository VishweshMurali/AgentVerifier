"""Shared model I/O for the LLM arms: the prompt-size guard and the deterministic answer parser.

There is no fuzzy matching. An answer either resolves to exactly one label or it is `None`, which
scores as wrong. `LLMUnavailable` and `LLMUsageMissing` are never caught here: they stop the run.
"""

import json
from typing import Any

from agent_committee.arms.config import MAX_PROMPT_CHARS
from agent_committee.arms.labels import LabelBook
from agent_committee.llm import LLMClient, LLMResult


class PromptTooLong(RuntimeError):
    """A prompt would exceed the context cap, where Ollama silently drops its head."""


def prompt_chars(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    """Characters sent to the model, counting tool schemas and replayed tool calls."""
    total = len(json.dumps(tools)) if tools else 0
    for message in messages:
        total += len(message.get("content") or "")
        if message.get("tool_calls"):
            total += len(json.dumps(message["tool_calls"]))
    return total


def guarded_chat(
    client: LLMClient,
    messages: list[dict[str, Any]],
    *,
    arm: str,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
) -> LLMResult:
    """`client.chat`, refusing beforehand any prompt too long to reach the model intact."""
    size = prompt_chars(messages, tools)
    if size > MAX_PROMPT_CHARS:
        raise PromptTooLong(f"{size} chars exceeds MAX_PROMPT_CHARS={MAX_PROMPT_CHARS}")
    return client.chat(messages, arm=arm, tools=tools, max_tokens=max_tokens)


def parse_answer(result: LLMResult, book: LabelBook) -> str | None:
    """The canonical label, or None for empty, truncated, multi-line or unknown output."""
    text = result.content.strip()
    if result.truncated or not text or "\n" in text:
        return None
    return book.resolve(text)
