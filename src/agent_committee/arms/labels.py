"""Label set and label definitions derived deterministically from the pool.

No LLM and no hand-written text: each label gets a humanised name, its most discriminative
terms (class-mean TF-IDF minus global-mean TF-IDF) and its medoid utterances.
"""

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from agent_committee.schemas import Case

N_TERMS = 8
N_MEDOIDS = 3


@dataclass(frozen=True)
class LabelInfo:
    label: str
    name: str
    terms: tuple[str, ...]
    examples: tuple[str, ...]

    def definition(self) -> str:
        return (
            f"{self.label} ({self.name}). Key terms: {', '.join(self.terms)}. "
            f"Examples: " + " | ".join(f'"{e}"' for e in self.examples)
        )


def humanize(label: str) -> str:
    """'Refund_not_showing_up' -> 'refund not showing up'; 'reverted_card_payment?' -> ..."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", label.casefold()).split())


def label_key(text: str) -> str:
    """Case, punctuation and underscore-insensitive key used to match a model's answer."""
    return re.sub(r"[^0-9a-z]+", "", text.casefold())


@dataclass(frozen=True)
class LabelBook:
    infos: tuple[LabelInfo, ...]  # sorted by label

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(i.label for i in self.infos)

    def get(self, label: str) -> LabelInfo:
        for info in self.infos:
            if info.label == label:
                return info
        raise KeyError(label)

    def resolve(self, text: str) -> str | None:
        """Map a model's answer to exactly one canonical label, or None."""
        key = label_key(text)
        hits = [i.label for i in self.infos if label_key(i.label) == key]
        return hits[0] if len(hits) == 1 else None


def build_label_book(pool: list[Case]) -> LabelBook:
    if not pool:
        raise ValueError("empty pool")
    cases = sorted(pool, key=lambda c: c.id)
    labels = sorted({c.label for c in cases})
    keys = [label_key(label) for label in labels]
    if len(set(keys)) != len(keys):
        raise ValueError("labels collide after normalisation")
    vec = TfidfVectorizer(stop_words="english", lowercase=True)
    x = vec.fit_transform([c.text for c in cases])
    terms = np.array(vec.get_feature_names_out())
    global_mean = np.asarray(x.mean(axis=0)).ravel()
    infos = []
    for label in labels:
        idx = np.array([i for i, c in enumerate(cases) if c.label == label])
        xc = x[idx]
        diff = np.asarray(xc.mean(axis=0)).ravel() - global_mean
        # stable: highest score first, ties broken alphabetically
        order = sorted(range(len(terms)), key=lambda j: (-diff[j], terms[j]))
        top = tuple(str(terms[j]) for j in order[:N_TERMS])
        sim = (xc @ xc.T).toarray()
        n = len(idx)
        mean_sim = (sim.sum(axis=1) - np.diag(sim)) / max(n - 1, 1)
        ranked = sorted(range(n), key=lambda i: (-mean_sim[i], cases[idx[i]].id))
        ex = tuple(cases[idx[i]].text for i in ranked[:N_MEDOIDS])
        infos.append(LabelInfo(label=label, name=humanize(label), terms=top, examples=ex))
    return LabelBook(infos=tuple(infos))
