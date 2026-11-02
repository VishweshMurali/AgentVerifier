"""The agent's two tools: search labelled pool examples, read one label's definition.

`search_examples` is the single swappable search. It is backed by the linear layer's shortlist for
the model's own query, so the model chooses what to look for and a better backend can replace it.
"""

import json
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from agent_committee.arms.labels import LabelBook
from agent_committee.arms.linear import LinearLayer
from agent_committee.arms.retrieval import Retriever

SEARCH = "search_examples"
READ_LABEL = "read_label_definition"
EXAMPLES_PER_LABEL = 2
SEARCH_DEPTH = 300
N_CLOSEST_LABELS = 3

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": SEARCH,
            "description": "Find likely labels and labelled banking queries for your search words.",
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
    def __init__(self, retriever: Retriever, book: LabelBook, layer: LinearLayer) -> None:
        self._retriever = retriever
        self._book = book
        self._layer = layer
        self._names = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
        self._name_matrix = self._names.fit_transform([info.name for info in book.infos])

    def search_examples(self, query: str) -> str:
        """Closest pool examples for each label the layer shortlists, best label first."""
        shortlist = self._layer.decide(query).shortlist
        found: dict[str, list[str]] = {label: [] for label in shortlist}
        for case, _ in self._retriever.nearest(query, SEARCH_DEPTH):
            if case.label in found and len(found[case.label]) < EXAMPLES_PER_LABEL:
                found[case.label].append(case.text)
        lines = []
        for label in shortlist:
            texts = found[label] or list(self._book.get(label).examples[:EXAMPLES_PER_LABEL])
            lines.extend(f"{text} => {label}" for text in texts)
        return "\n".join(lines)

    def closest_labels(self, text: str) -> list[str]:
        """Real labels nearest to a bad guess, by character overlap. Ties keep label order."""
        scores = (self._name_matrix @ self._names.transform([text]).T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:N_CLOSEST_LABELS]
        return [self._book.infos[i].label for i in order]

    def read_label(self, label: str) -> str:
        canonical = self._book.resolve(label)
        if canonical is None:
            return (
                f"Unknown label: {label}. Closest labels: {', '.join(self.closest_labels(label))}"
            )
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
