"""
evaluate.py
-----------
Functional Module 3: Evaluation, Reporting & Analytics.

Responsibilities:
    - Score every trained model on the held-out test set.
    - Compute accuracy, precision, recall, F1, and confusion matrix.
    - Produce a human-readable comparison table.
    - Save a JSON analytics report to disk for later inspection.

Input:  dict of trained models + test features/labels
Output: dict of per-model metrics, and a saved JSON report
"""

import json
import os
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from src import config
from src.logger import get_logger

logger = get_logger(__name__)


def evaluate_models(trained_models: dict, X_test, y_test) -> dict:
    """Evaluate each trained model and return a metrics dictionary."""
    results = {}
    for name, clf in trained_models.items():
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "confusion_matrix": cm.tolist(),
        }
        results[name] = metrics
        logger.info("Evaluated %s -> accuracy=%.4f, f1=%.4f", name, metrics["accuracy"], metrics["f1_score"])
    return results


def print_comparison_table(results: dict) -> None:
    """Print a simple aligned comparison table to the console."""
    header = f"{'Model':<22}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-score':<12}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<22}{m['accuracy']:<12}{m['precision']:<12}{m['recall']:<12}{m['f1_score']:<12}")


def save_report(results: dict, path: str = None) -> str:
    """Persist evaluation results as a timestamped JSON report."""
    path = path or os.path.join(
        config.REPORT_DIR, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation report to %s", path)
    return path


def best_model(results: dict) -> str:
    """Return the name of the model with the highest F1-score."""
    return max(results, key=lambda name: results[name]["f1_score"])
