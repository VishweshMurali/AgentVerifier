"""The shared first step and arm 1: word + character logistic regression.

Trained once on the pool and then frozen. Prompt and agent read its shortlist; the linear arm
answers from it alone. Gold labels reach `fit` only; `decide` sees text.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from agent_committee.arms import config
from agent_committee.schemas import ArmName, ArmPrediction, Case

SEED = 0
C = 20.0
FINGERPRINT_DECIMALS = 6  # coefficients are floats; compare at a precision a rebuild reproduces


@dataclass(frozen=True)
class Decision:
    """`score` is the top label's probability. `null` flags a low score; it never scores a case."""

    choice: str
    score: float
    null: bool
    shortlist: tuple[str, ...]


class LinearLayer:
    def __init__(
        self,
        threshold: float = config.NULL_THRESHOLD,
        shortlist_k: int = config.SHORTLIST_K,
    ) -> None:
        self.threshold = threshold
        self.shortlist_k = shortlist_k
        self._word = TfidfVectorizer(lowercase=True, sublinear_tf=True)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._model = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, pool: list[Case]) -> None:
        if self._fitted:
            raise RuntimeError("layer is frozen; it can be fitted only once")
        cases = sorted(pool, key=lambda c: c.id)
        texts = [c.text for c in cases]
        features = hstack([self._word.fit_transform(texts), self._char.fit_transform(texts)])
        self._model.fit(features.tocsr(), [c.label for c in cases])
        self._fitted = True

    def decide(self, text: str) -> Decision:
        self._require_fitted()
        features = hstack([self._word.transform([text]), self._char.transform([text])])
        probabilities = self._model.predict_proba(features.tocsr())[0]
        # classes_ is sorted, so a stable sort breaks ties by label name
        order = np.argsort(-probabilities, kind="stable")[: self.shortlist_k]
        labels = tuple(str(self._model.classes_[i]) for i in order)
        score = float(probabilities[order[0]])
        return Decision(
            choice=labels[0],
            score=score,
            null=score < self.threshold,
            shortlist=labels,
        )

    def fingerprint(self) -> str:
        """SHA-256 of all that fixes the layer's output. Equal after a rebuild from the pool."""
        self._require_fitted()
        digest = hashlib.sha256()
        digest.update(repr((self.threshold, self.shortlist_k)).encode())
        digest.update("\n".join(str(c) for c in self._model.classes_).encode())
        for vectorizer in (self._word, self._char):
            digest.update("\n".join(vectorizer.get_feature_names_out()).encode())
        for values in (self._model.coef_, self._model.intercept_):
            # adding 0.0 turns -0.0 into 0.0, so rounding noise near zero cannot change the bytes
            digest.update((np.round(values, FINGERPRINT_DECIMALS) + 0.0).tobytes())
        return digest.hexdigest()

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit() first")


def make_layer(pool: list[Case]) -> LinearLayer:
    layer = LinearLayer()
    layer.fit(pool)
    return layer


class LinearArm:
    """Arm 1: the layer's choice with no LLM. It always answers, flagged or not."""

    name = ArmName.LINEAR

    def __init__(self, layer: LinearLayer) -> None:
        self._layer = layer

    def fit(self, pool: list[Case]) -> None:
        if not self._layer.fitted:
            self._layer.fit(pool)

    def predict(self, case_id: str, text: str) -> ArmPrediction:
        choice = self._layer.decide(text).choice
        # no LLM tokens; local latency is negligible and not clocked inside arms/
        return ArmPrediction(case_id=case_id, arm=self.name, pred=choice)
