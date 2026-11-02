"""TF-IDF cosine nearest neighbours over the pool. Deterministic: ties break by case id."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from agent_committee.schemas import Case


class Retriever:
    def __init__(self, pool: list[Case]) -> None:
        if not pool:
            raise ValueError("empty pool")
        self._pool = sorted(pool, key=lambda c: c.id)
        self._vec = TfidfVectorizer(lowercase=True, sublinear_tf=True)
        self._x = self._vec.fit_transform([c.text for c in self._pool])

    def __len__(self) -> int:
        return len(self._pool)

    def nearest(self, text: str, k: int = 5) -> list[tuple[Case, float]]:
        """Top-k pool cases by cosine similarity, highest first, ties by ascending case id."""
        if k <= 0:
            return []
        sims = (self._x @ self._vec.transform([text]).T).toarray().ravel()
        # pool is sorted by id, so a stable sort on -sim breaks ties by id
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self._pool[i], float(sims[i])) for i in order]
