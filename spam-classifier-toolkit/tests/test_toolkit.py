"""
test_toolkit.py
----------------
Basic unit/validation tests for the toolkit. Run with:
    pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, load_dataset, preprocess
from src.model import SpamClassifier
from src.exceptions import DatasetError, ModelNotFoundError, ModelNotTrainedError
from src import config


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("HELLO!!! Call NOW 123") == "hello call now"


def test_clean_text_rejects_non_string():
    with pytest.raises(DatasetError):
        clean_text(12345)


def test_load_dataset_missing_file_raises():
    with pytest.raises(DatasetError):
        load_dataset("data/does_not_exist.csv")


def test_load_dataset_missing_columns_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(bad_csv, index=False)
    with pytest.raises(DatasetError):
        load_dataset(str(bad_csv))


def test_preprocess_end_to_end(tmp_path):
    df = pd.DataFrame({
        config.LABEL_COLUMN: ["spam", "ham"] * 20,
        config.TEXT_COLUMN: ["win free prize now call"] * 20 + ["let's meet for lunch"] * 20,
    })
    X_train, X_test, y_train, y_test, vec = preprocess(df)
    assert X_train.shape[0] + X_test.shape[0] == 40
    assert X_train.shape[1] == X_test.shape[1]


def test_model_unknown_name_raises():
    with pytest.raises(ModelNotFoundError):
        SpamClassifier("not_a_real_model")


def test_model_predict_before_train_raises():
    clf = SpamClassifier("naive_bayes")
    with pytest.raises(ModelNotTrainedError):
        clf.predict([[0, 0, 1]])


def test_model_train_and_predict_shapes():
    df = pd.DataFrame({
        config.LABEL_COLUMN: ["spam", "ham"] * 20,
        config.TEXT_COLUMN: ["win free prize now call"] * 20 + ["let's meet for lunch"] * 20,
    })
    X_train, X_test, y_train, y_test, vec = preprocess(df)
    clf = SpamClassifier("naive_bayes").train(X_train, y_train)
    preds = clf.predict(X_test)
    assert len(preds) == X_test.shape[0]
