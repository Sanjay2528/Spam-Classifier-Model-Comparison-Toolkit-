# Spam Classifier & Model Comparison Toolkit

## Overview
A command-line toolkit that trains and compares three classic machine-learning
models (Naive Bayes, Logistic Regression, Random Forest) for classifying
SMS/email text messages as **spam** or **ham** (not spam). Built for the
"Fundamentals of AI and ML" course project, demonstrating text preprocessing,
feature engineering (TF-IDF), model training, and automated evaluation.

## Features
- **Data preprocessing module** — cleans raw text and converts it into TF-IDF
  numeric features, with validation of the input dataset.
- **Model training module** — trains and wraps three interchangeable ML
  classifiers behind one consistent interface.
- **Evaluation & reporting module** — computes accuracy, precision, recall,
  F1-score, and a confusion matrix for every model, prints a comparison
  table, and saves a timestamped JSON report.
- **Ad-hoc prediction** — classify any single message from the command line
  using the best-performing model.
- Centralized configuration, structured logging (console + file), and a
  custom exception hierarchy for clear error handling.

## Technologies / Tools Used
- Python 3.10+
- pandas, scikit-learn
- pytest (unit testing)
- argparse (CLI)

## Project Structure
```
spam-classifier-toolkit/
├── src/
│   ├── config.py         # paths & tunable parameters
│   ├── logger.py         # centralized logging
│   ├── exceptions.py     # custom exception hierarchy
│   ├── preprocessing.py  # Module 1: data cleaning & feature extraction
│   ├── model.py          # Module 2: model training & prediction
│   ├── evaluate.py       # Module 3: evaluation & reporting
│   └── main.py           # CLI entry point
├── scripts/
│   └── generate_sample_data.py  # generates the sample dataset
├── tests/
│   └── test_toolkit.py   # unit tests
├── data/
│   └── sample_dataset.csv
├── reports/               # generated evaluation reports (JSON)
├── logs/                  # generated log files
├── requirements.txt
├── statement.md
└── README.md
```

## Steps to Install & Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sanjay2528/Spam-Classifier-Model-Comparison-Toolkit-.git
   cd Spam-Classifier-Model-Comparison-Toolkit-
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate the sample dataset** (only needed once; a dataset is already
   included, but you can regenerate/replace it)
   ```bash
   python3 scripts/generate_sample_data.py
   ```

5. **Run the toolkit**
   ```bash
   python3 -m src.main
   ```
   This trains all three models, prints a comparison table, and saves a
   JSON evaluation report under `reports/`.

6. **Classify a custom message**
   ```bash
   python3 -m src.main --message "Congratulations! You won a free prize, call now"
   ```

7. **Use your own dataset**
   ```bash
   python3 -m src.main --data path/to/your_dataset.csv
   ```
   The CSV must contain two columns: `label` (spam/ham) and `message` (text).

## Instructions for Testing
Run the automated unit test suite with:
```bash
pytest tests/ -v
```
This covers text cleaning, dataset validation/error handling, the
preprocessing pipeline, and model training/prediction (including expected
exceptions for invalid input).

## Notes
- The bundled `data/sample_dataset.csv` is synthetically generated
  (see `scripts/generate_sample_data.py`) so the project runs end-to-end
  with no external downloads. For more realistic results, swap in a larger
  public dataset (e.g. the UCI SMS Spam Collection) using the same two
  columns.

  ## Screenshots

### Model Training & Evaluation
<img width="1620" height="900" alt="architecture" src="https://github.com/user-attachments/assets/c8df9818-071e-42ee-9575-3a93e6157ed1" />
<img width="1620" height="936" alt="class_diagram" src="https://github.com/user-attachments/assets/265482c4-4d53-4d6f-86e2-960c84973275" />
<img width="1620" height="990" alt="sequence" src="https://github.com/user-attachments/assets/fc8ee1ed-440b-47ac-8ab3-21a381d6a8a0" />
<img width="1260" height="900" alt="usecase" src="https://github.com/user-attachments/assets/fa6dd54b-c0d4-4538-8390-572509dc5aaa" />
<img width="1620" height="576" alt="workflow" src="https://github.com/user-attachments/assets/64945bdc-c574-460b-a40b-70651bec5209" />





