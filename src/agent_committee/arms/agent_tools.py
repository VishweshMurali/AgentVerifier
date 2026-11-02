"""The agent's two tools: retrieve similar pool examples, read one label's definition."""

import json
from typing import Any

from agent_committee.arms.config import RETRIEVAL_K
from agent_committee.arms.labels import LabelBook
from agent_committee.arms.retrieval import Retriever

SEARCH = "search_examples"
READ_LABEL = "read_label_definition"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": SEARCH,
            "description": "Find labelled banking queries similar to the given query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": READ_LABEL,
            "description": "Read the definition of one intent label.",
            "parameters": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
        },
    },
]


class ToolBox:
    def __init__(self, retriever: Retriever, book: LabelBook) -> None:
        self._retriever = retriever
        self._book = book

    def search_examples(self, query: str) -> str:
        hits = self._retriever.nearest(query, RETRIEVAL_K)
        return "\n".join(f"{case.text} => {case.label}" for case, _ in hits)

    def read_label(self, label: str) -> str:
        canonical = self._book.resolve(label)
        if canonical is None:
            return f"Unknown label: {label}"
        return self._book.get(canonical).definition()

    def run(self, name: str, arguments: str) -> str:
        """Tool output as text. Bad calls return an error the model can read and recover from."""
        try:
            args = json.loads(arguments)
        except (TypeError, ValueError):
            return f"Invalid JSON arguments for {name}"
        if not isinstance(args, dict):
            return f"Invalid arguments for {name}"
        if name == SEARCH and isinstance(args.get("query"), str):
            return self.search_examples(args["query"])
        if name == READ_LABEL and isinstance(args.get("label"), str):
            return self.read_label(args["label"])
        return f"Invalid call: {name}"
