"""Arm 1: TF-IDF + logistic regression. Fitted on the pool only."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from agent_committee.schemas import ArmName, ArmPrediction, Case

SEED = 0
C = 20.0


class TfidfArm:
    name = ArmName.TFIDF

    def __init__(self) -> None:
        self._model = make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(C=C, max_iter=2000, random_state=SEED),
        )
        self._fitted = False

    def fit(self, pool: list[Case]) -> None:
        cases = sorted(pool, key=lambda c: c.id)
        self._model.fit([c.text for c in cases], [c.label for c in cases])
        self._fitted = True

    def predict(self, case_id: str, text: str) -> ArmPrediction:
        if not self._fitted:
            raise RuntimeError("fit() first")
        pred = str(self._model.predict([text])[0])
        # no LLM tokens; local latency is negligible and not clocked inside arms/
        return ArmPrediction(case_id=case_id, arm=self.name, pred=pred)
