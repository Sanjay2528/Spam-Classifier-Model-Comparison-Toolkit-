"""
model.py
--------
Functional Module 2: Model Training & Prediction.

Wraps three classic ML algorithms behind one consistent interface so
they can be trained and compared interchangeably:
    - Multinomial Naive Bayes
    - Logistic Regression
    - Random Forest

Input:  training features/labels
Output: a trained model object that exposes predict() and predict_proba()
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.exceptions import ModelNotFoundError, ModelNotTrainedError
from src.logger import get_logger
from src import config

logger = get_logger(__name__)

_MODEL_REGISTRY = {
    "naive_bayes": lambda: MultinomialNB(),
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=config.RANDOM_STATE
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=150, random_state=config.RANDOM_STATE, n_jobs=-1
    ),
}


class SpamClassifier:
    """Thin, uniform wrapper around a scikit-learn classifier."""

    def __init__(self, model_name: str):
        if model_name not in _MODEL_REGISTRY:
            raise ModelNotFoundError(
                f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY)}"
            )
        self.model_name = model_name
        self._model = _MODEL_REGISTRY[model_name]()
        self._is_trained = False

    def train(self, X_train, y_train):
        logger.info("Training model: %s", self.model_name)
        self._model.fit(X_train, y_train)
        self._is_trained = True
        return self

    def predict(self, X):
        if not self._is_trained:
            raise ModelNotTrainedError(f"{self.model_name} must be trained before predicting")
        return self._model.predict(X)

    def predict_proba(self, X):
        if not self._is_trained:
            raise ModelNotTrainedError(f"{self.model_name} must be trained before predicting")
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None


def train_all_models(X_train, y_train, model_names=None):
    """Train every model listed in config.MODEL_NAMES (or a custom subset)."""
    model_names = model_names or config.MODEL_NAMES
    trained = {}
    for name in model_names:
        clf = SpamClassifier(name).train(X_train, y_train)
        trained[name] = clf
    return trained
