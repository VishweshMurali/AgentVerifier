"""Classic rungs of the comparator ladder (v2 P3): no LLM, fitted on the pool only, deterministic.

- MajorityArm: always the pool's most common label (ties by sorted label). The accuracy floor any
  other arm must clear.
- NaiveBayesArm: multinomial Naive Bayes on word 1-2 grams and character 2-4 grams. A cheap
  rung different in kind from the shared logistic-regression layer.

Tier 1 and tier 2 rungs (owner approved 2026-11-10), all on word 1-2 + character 2-4 gram TF-IDF:
- LinearSVMArm: a linear support vector machine (tier 1, a strong simple text classifier).
- RandomForestArm: bagging, a random forest (tier 2).
- BoostingArm: gradient boosting (histogram trees, tier 2).
The two tree ensembles first compress the TF-IDF features to SVD_DIM dimensions (trees cannot use
hundreds of thousands of sparse columns at this cost). Settings are fixed constants, chosen before
any score was seen, except BOOST_L2: added after the first Banking77 DEV pass showed boosting
diverging, picked from training-fit stability only (one value, never tuned on a score). Every
random step has a fixed seed and the pool is sorted by id, so the runner's
refit at test time reproduces the DEV fit.

All cost no tokens; non-LLM rungs are priced at $0 (owner decision 2026-11-09).
"""

from collections import Counter

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.svm import LinearSVC

from agent_committee.schemas import ArmName, ArmPrediction, Case

NB_ALPHA = 1.0
SEED = 0
SVM_C = 1.0
SVD_DIM = 300
FOREST_TREES = 200
BOOST_ROUNDS = 100
BOOST_LEARNING_RATE = 0.1
# without it the many-class fit diverges (scores in the 100,000s after 20 rounds on Banking77's
# pool, training accuracy 13%); with it training accuracy is 100% and scores stay small
BOOST_L2 = 1.0


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


def _tfidf() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
        ]
    )


class _Compress(TruncatedSVD):
    """TruncatedSVD that never asks for more dimensions than the data has (tiny pools)."""

    def fit_transform(self, X, y=None):
        self.n_components = max(1, min(SVD_DIM, min(X.shape) - 1))
        return super().fit_transform(X, y)


class _SklearnArm:
    """Fit on the pool sorted by id; answer with the highest-scoring class (ties: sorted label)."""

    name: ArmName

    def __init__(self) -> None:
        self._model: Pipeline | None = None

    def _pipeline(self) -> Pipeline:
        raise NotImplementedError

    def fit(self, pool: list[Case]) -> None:
        ordered = sorted(pool, key=lambda c: c.id)
        self._model = self._pipeline()
        self._model.fit([c.text for c in ordered], [c.label for c in ordered])

    def _scores(self, text: str) -> np.ndarray:
        assert self._model is not None
        return self._model.predict_proba([text])[0]

    def predict(self, case_id: str, text: str) -> ArmPrediction:
        if self._model is None:
            raise RuntimeError("fit() first")
        # argmax takes the first maximum, and classes_ are sorted: ties go to the sorted label
        label = str(self._model.classes_[int(np.argmax(self._scores(text)))])
        return ArmPrediction(case_id=case_id, arm=self.name, pred=label)


class LinearSVMArm(_SklearnArm):
    name = ArmName.LINEAR_SVM

    def _pipeline(self) -> Pipeline:
        return make_pipeline(_tfidf(), LinearSVC(C=SVM_C, random_state=SEED))

    def _scores(self, text: str) -> np.ndarray:
        assert self._model is not None
        scores = np.ravel(self._model.decision_function([text]))
        if len(self._model.classes_) == 2:  # one margin: positive means the second class
            return np.array([-scores[0], scores[0]])
        return scores


class RandomForestArm(_SklearnArm):
    name = ArmName.RANDOM_FOREST

    def _pipeline(self) -> Pipeline:
        forest = RandomForestClassifier(n_estimators=FOREST_TREES, random_state=SEED, n_jobs=1)
        return make_pipeline(_tfidf(), _Compress(random_state=SEED), forest)


class BoostingArm(_SklearnArm):
    name = ArmName.BOOSTING

    def _pipeline(self) -> Pipeline:
        boost = HistGradientBoostingClassifier(
            max_iter=BOOST_ROUNDS,
            learning_rate=BOOST_LEARNING_RATE,
            l2_regularization=BOOST_L2,
            early_stopping=False,  # its validation split would be another random step
            random_state=SEED,
        )
        return make_pipeline(_tfidf(), _Compress(random_state=SEED), boost)
