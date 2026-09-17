"""
config.py
---------
Central configuration for the Spam Classifier & Model Comparison Toolkit.
Keeping all tunable parameters in one place makes the system easier to
maintain and reduces the risk of "magic numbers" scattered across modules.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

RAW_DATASET_PATH = os.path.join(DATA_DIR, "sample_dataset.csv")
LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
TEXT_COLUMN = "message"
LABEL_COLUMN = "label"
TEST_SIZE = 0.25
RANDOM_STATE = 42
MAX_FEATURES = 3000  # cap on TF-IDF vocabulary size (controls memory/perf)

# ---------------------------------------------------------------------------
# Models to train and compare
# ---------------------------------------------------------------------------
MODEL_NAMES = ["naive_bayes", "logistic_regression", "random_forest"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"

# Ensure required directories exist at import time so downstream modules
# never have to worry about missing folders.
for _dir in (DATA_DIR, LOG_DIR, REPORT_DIR):
    os.makedirs(_dir, exist_ok=True)
