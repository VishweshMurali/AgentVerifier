"""Classic rungs of the comparator ladder (v2 P3): no LLM, fitted on the pool only, deterministic.

- MajorityArm: always the pool's most common label (ties by sorted label). The accuracy floor any
  other arm must clear.
- NaiveBayesArm: multinomial Naive Bayes on word 1-2 grams and character 2-4 grams. A cheap
  rung different in kind from the shared logistic-regression layer.

Both cost no tokens; non-LLM rungs are priced at $0 (owner decision 2026-11-09).
"""

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, make_pipeline

from agent_committee.schemas import ArmName, ArmPrediction, Case

NB_ALPHA = 1.0


class MajorityArm:
    name = ArmName.MAJORITY

    def __init__(self) -> None:
        self._label: str | None = None

    def fit(self, pool: list[Case]) -> None:
        counts = Counter(c.label for c in pool)
        top = max(counts.values())
        self._label = sorted(label for label, n in counts.items() if n == top)[0]

    def predict(self, case_id: str, text: str) -> ArmPrediction:
        if self._label is None:
            raise RuntimeError("fit() first")
        return ArmPrediction(case_id=case_id, arm=self.name, pred=self._label)


class NaiveBayesArm:
    name = ArmName.NAIVE_BAYES

    def __init__(self) -> None:
        self._model = None

    def fit(self, pool: list[Case]) -> None:
        features = FeatureUnion(
            [
                ("word", CountVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)),
                ("char", CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)),
            ]
        )
        ordered = sorted(pool, key=lambda c: c.id)
        self._model = make_pipeline(features, MultinomialNB(alpha=NB_ALPHA))
        self._model.fit([c.text for c in ordered], [c.label for c in ordered])

    def predict(self, case_id: str, text: str) -> ArmPrediction:
        if self._model is None:
            raise RuntimeError("fit() first")
        scores = self._model.predict_log_proba([text])[0]
        # argmax takes the first maximum, and classes_ are sorted: ties go to the sorted label
        label = str(self._model.classes_[int(np.argmax(scores))])
        return ArmPrediction(case_id=case_id, arm=self.name, pred=label)
