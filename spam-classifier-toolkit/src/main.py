"""
main.py
-------
Command-line entry point for the Spam Classifier & Model Comparison Toolkit.
Ties together preprocessing -> training -> evaluation -> reporting.

Usage:
    python -m src.main
    python -m src.main --data data/sample_dataset.csv
    python -m src.main --message "Congratulations! You won a free prize, call now"
"""

import argparse

from src import config
from src.preprocessing import load_dataset, preprocess, clean_text
from src.model import train_all_models
from src.evaluate import evaluate_models, print_comparison_table, save_report, best_model
from src.exceptions import SpamToolkitError
from src.logger import get_logger

logger = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare spam-classification models from the command line."
    )
    parser.add_argument(
        "--data", default=config.RAW_DATASET_PATH,
        help="Path to the training CSV (columns: label, message)."
    )
    parser.add_argument(
        "--message", default=None,
        help="Optional: classify a single ad-hoc message with the best model after training."
    )
    return parser


def run(data_path: str, adhoc_message: str = None) -> None:
    try:
        df = load_dataset(data_path)
        X_train, X_test, y_train, y_test, vectorizer = preprocess(df)

        trained_models = train_all_models(X_train, y_train)
        results = evaluate_models(trained_models, X_test, y_test)

        print_comparison_table(results)
        report_path = save_report(results)
        top = best_model(results)
        print(f"\nBest performing model: {top} (F1-score = {results[top]['f1_score']})")
        print(f"Full report saved to: {report_path}")

        if adhoc_message:
            cleaned = clean_text(adhoc_message)
            vec = vectorizer.transform([cleaned])
            prediction = trained_models[top].predict(vec)[0]
            print(f"\nMessage: {adhoc_message!r}\nPredicted label ({top}): {prediction}")

    except SpamToolkitError as exc:
        logger.error("Application error: %s", exc)
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args.data, args.message)
