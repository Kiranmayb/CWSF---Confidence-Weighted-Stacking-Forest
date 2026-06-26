import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
def shannon_entropy(probs, epsilon=1e-12):
    probs = np.clip(probs, epsilon, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def entropy_to_confidence(entropy, epsilon=1e-6):
    return 1.0 / (entropy + epsilon)


def normalize_weights(weight_matrix):
    row_sums = weight_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid div-by-zero
    return weight_matrix / row_sums


def compute_model_weights(proba_list, reliability_scores=None):
    entropies = np.column_stack([
        shannon_entropy(p) for p in proba_list
    ])  # shape (n_samples, n_models)

    confidences = entropy_to_confidence(entropies)

    if reliability_scores is not None:
        reliability_scores = np.asarray(reliability_scores)
        confidences = confidences * reliability_scores[np.newaxis, :]

    return normalize_weights(confidences)

class CWSFClassifier(BaseEstimator, ClassifierMixin):
    """
    Well, this is a Confidence-Weighted Stacking Forest classifier, called as CWSF....

    It has 6 Parameters which are :
    ----------
    base_estimators : list of (str, estimator) tuples, optional
        Base models to ensemble. Each must support predict_proba().
        If None, defaults to [RandomForest, LogisticRegression, KNN] --
        three models with very different decision boundaries, which
        is what makes confidence-weighting meaningful (they disagree
        in different regions of feature space).
    random_state : int, optional
        For reproducibility.
    use_reliability_weighting : bool, default=True
        If True, multiplies entropy-based confidence by each model's
        held-out validation accuracy. Down-weights models that are
        globally unreliable even when locally overconfident.
    use_calibration : bool, default=True
        If True, wraps each base estimator in CalibratedClassifierCV
        (isotonic regression) before computing entropy. This is the
        primary fix for the "confidently wrong" problem: a model like
        KNN can output a sharply peaked but poorly-calibrated
        probability (e.g. [0.97, 0.03] when it is actually wrong 30%
        of the time at that confidence level). Calibration rescales
        predict_proba outputs so that "0.9 confidence" really does
        mean "right about 90% of the time" on held-out data, which
        makes the entropy values comparable and trustworthy ACROSS
        different model types -- not just within one model's own
        scale. This fixes the root cause; reliability weighting
        (above) is now a secondary safeguard rather than the main fix.
    validation_size : float, default=0.2
        Fraction of training data held out internally to measure
        reliability scores.
    calibration_cv : int, default=3
        Number of cross-validation folds used internally by
        CalibratedClassifierCV to fit the calibration curve.

    It hs 4 Attributes which are :
    ----------
    estimators_ : list of fitted (and optionally calibrated) base estimators
    classes_ : ndarray of class labels seen during fit
    n_features_in_ : int
    reliability_scores_ : ndarray of shape (n_models,) or None

    It is an example if you want to test it with a sci-kit dataset :
    -------
    >>> from cwsf_model import CWSFClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=500, n_classes=3, n_informative=5, random_state=0)
    >>> model = CWSFClassifier()
    >>> model.fit(X, y)
    >>> model.predict(X[:5])
    """

    def __init__(self, base_estimators=None, random_state=None,
                use_reliability_weighting=True, use_calibration=True,
                validation_size=0.2, calibration_cv=3):
        self.base_estimators = base_estimators
        self.random_state = random_state
        self.use_reliability_weighting = use_reliability_weighting
        self.use_calibration = use_calibration
        self.validation_size = validation_size
        self.calibration_cv = calibration_cv

    def _default_estimators(self):
        return [
            ("rf", RandomForestClassifier(
                n_estimators=150, random_state=self.random_state)),
            ("logreg", LogisticRegression(
                max_iter=1000, random_state=self.random_state)),
            ("knn", KNeighborsClassifier(n_neighbors=7)),
        ]

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        estimators = self.base_estimators or self._default_estimators()
        self.estimator_names_ = [name for name, _ in estimators]

        def _build_model(est):
        #I have used CallibratedClassifierCV as this will actually map the predicted probability and the true label and generate a isotonic curve because it will be more flexible on enough samples and can avoid overfitting.
            if self.use_calibration:
                return CalibratedClassifierCV(
                    clone(est), method="isotonic", cv=self.calibration_cv
                )
            return clone(est)
        if self.use_reliability_weighting:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X, y, test_size=self.validation_size,
                random_state=self.random_state, stratify=y
            )

            measuring_models = []
            for name, est in estimators:
                model = _build_model(est)
                model.fit(X_fit, y_fit)
                measuring_models.append(model)

            self.reliability_scores_ = np.array([
                accuracy_score(y_val, m.predict(X_val))
                for m in measuring_models
            ])
        else:
            self.reliability_scores_ = None
        self.estimators_ = []
        for name, est in estimators:
            model = _build_model(est)
            model.fit(X, y)
            self.estimators_.append(model)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)

        proba_list = [est.predict_proba(X) for est in self.estimators_]

        # Here is the actual twist I mean there is Per-sample, per-model confidence weight. The CWSF model will judge this and will choose the best model among its base estimators with respect to the confidence. 
        weights = compute_model_weights(
            proba_list, reliability_scores=self.reliability_scores_
        )

        combined = np.zeros_like(proba_list[0])
        for i, proba in enumerate(proba_list):
            combined += weights[:, i][:, np.newaxis] * proba

        combined = combined / combined.sum(axis=1, keepdims=True)
        return combined

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_model_weights(self, X):
        """
        It is a Diagnostic method (not part of standard sklearn API):
        returns the raw per-sample, per-model confidence weights.
        Useful for understanding WHY the model made a decision --
        e.g. "for this sample, RF was trusted 70% and KNN only 20%."

        Returns
        -------
        weights : ndarray of shape (n_samples, n_models)
        names : list of str, model names in column order
        """
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        proba_list = [est.predict_proba(X) for est in self.estimators_]
        weights = compute_model_weights(
            proba_list, reliability_scores=self.reliability_scores_
        )
        return weights, self.estimator_names_


__version__ = "0.1.0"
"""
Hope this model increases the evaluation score of certain inputs, and you are always to collab

"""
