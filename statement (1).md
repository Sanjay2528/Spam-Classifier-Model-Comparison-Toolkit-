# Problem Statement

Unsolicited spam messages (SMS and email) are a persistent nuisance and
security risk — they range from advertising to phishing and fraud attempts.
Manually filtering these messages does not scale, so an automated,
content-based classifier is needed to flag likely-spam messages before a
user ever sees them.

## Scope of the Project
This project builds a command-line toolkit that:
- Cleans and vectorizes raw text messages using TF-IDF.
- Trains multiple supervised ML classifiers (Naive Bayes, Logistic
  Regression, Random Forest) on labelled spam/ham data.
- Evaluates and compares the models on a held-out test set using standard
  classification metrics.
- Allows a user to classify a new, unseen message from the command line
  using the best-performing model.

Out of scope: a graphical user interface, real-time email-server
integration, and deep-learning/transformer-based models (kept to classical
ML as covered in the course syllabus).

## Target Users
- Students/instructors evaluating the project against the course rubric.
- Developers who want a lightweight, extensible starting point for a text
  classification pipeline (e.g. to plug into an email client or chat app).

## High-Level Features
1. **Data preprocessing module** — validates and cleans raw text, extracts
   TF-IDF features, splits data into train/test sets.
2. **Model training & prediction module** — trains and compares three
   classical ML algorithms behind one shared interface.
3. **Evaluation & reporting module** — computes accuracy, precision,
   recall, F1-score and a confusion matrix; prints a comparison table and
   persists a JSON report; supports ad-hoc single-message classification.
