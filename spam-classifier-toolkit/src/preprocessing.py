"""
preprocessing.py
-----------------
Functional Module 1: Data Input & Preprocessing.

Responsibilities:
    - Load the raw dataset from disk (CSV).
    - Validate that it has the expected structure.
    - Clean the raw text (lowercasing, punctuation/number stripping).
    - Convert text into numeric features (TF-IDF) usable by ML models.
    - Split the data into train/test sets.

Input:  path to a CSV file with columns ["label", "message"]
Output: (X_train, X_test, y_train, y_test, fitted_vectorizer)
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src import config
from src.exceptions import DatasetError
from src.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits/extra whitespace from a message."""
    if not isinstance(text, str):
        raise DatasetError(f"Expected string message, got {type(text)}")
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str = config.RAW_DATASET_PATH) -> pd.DataFrame:
    """Load and validate the raw dataset."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DatasetError(f"Dataset not found at {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"Dataset at {path} is empty") from exc

    required_cols = {config.LABEL_COLUMN, config.TEXT_COLUMN}
    if not required_cols.issubset(df.columns):
        raise DatasetError(
            f"Dataset must contain columns {required_cols}, found {list(df.columns)}"
        )

    df = df.dropna(subset=[config.TEXT_COLUMN, config.LABEL_COLUMN])
    if df.empty:
        raise DatasetError("Dataset has no valid rows after dropping missing values")

    logger.info("Loaded dataset with %d rows from %s", len(df), path)
    return df


def preprocess(df: pd.DataFrame):
    """Clean text and produce TF-IDF features + train/test split."""
    df = df.copy()
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=config.MAX_FEATURES, stop_words="english")
    X = vectorizer.fit_transform(df[config.TEXT_COLUMN])
    y = df[config.LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    logger.info(
        "Preprocessing complete: %d train rows, %d test rows, %d features",
        X_train.shape[0], X_test.shape[0], X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test, vectorizer
