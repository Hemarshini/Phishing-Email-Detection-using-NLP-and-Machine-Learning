"""
Phishing Email Detection — Model Training Script
Trains and evaluates Logistic Regression and Random Forest classifiers
on email text using a TF-IDF based NLP pipeline.

Author: Divvela Hemarshini
"""

import pandas as pd
import numpy as np
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
# NOTE: Update these column names to match your actual phishing_emails.csv.
# The most common public version of this dataset uses "Email Text" and
# "Email Type" as column names. Confirm with df.columns before running.
TEXT_COLUMN = "Email Text"
LABEL_COLUMN = "Email Type"

df = pd.read_csv("data/phishing_emails.csv")
print("Columns found:", list(df.columns))
print(df.head())


# ---------------------------------------------------------------------------
# 2. Clean and prepare data
# ---------------------------------------------------------------------------
df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
df = df.drop_duplicates(subset=[TEXT_COLUMN])

# Normalize labels to binary: 1 = phishing, 0 = legitimate
# Handles string labels like "Phishing Email" / "Safe Email" robustly,
# regardless of pandas' internal string dtype representation.
if pd.api.types.is_numeric_dtype(df[LABEL_COLUMN]):
    df["label"] = df[LABEL_COLUMN]
else:
    df["label"] = df[LABEL_COLUMN].astype(str).str.lower().str.contains("phish").astype(int)


def clean_text(text):
    """Basic text normalization: lowercase, strip punctuation and digits."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                      # remove digits
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)


# ---------------------------------------------------------------------------
# 3. NLP feature extraction — TF-IDF
# ---------------------------------------------------------------------------
# TF-IDF with English stop-word removal, unigrams + bigrams, capped vocabulary
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Class balance:\n{y.value_counts()}")


# ---------------------------------------------------------------------------
# 4. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# ---------------------------------------------------------------------------
# 5. Train and evaluate Random Forest
# ---------------------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_metrics = {
    "accuracy": accuracy_score(y_test, rf_predictions),
    "precision": precision_score(y_test, rf_predictions),
    "recall": recall_score(y_test, rf_predictions),
    "f1": f1_score(y_test, rf_predictions),
}

print("\n--- Random Forest ---")
for k, v in rf_metrics.items():
    print(f"{k.capitalize()}: {v:.4f}")
print(classification_report(y_test, rf_predictions, target_names=["Legitimate", "Phishing"]))

rf_cm = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(5, 4))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"])
plt.title("Random Forest — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("results/random_forest_confusion_matrix.png")
plt.close()


# ---------------------------------------------------------------------------
# 6. Train and evaluate Logistic Regression
# ---------------------------------------------------------------------------
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

lr_metrics = {
    "accuracy": accuracy_score(y_test, lr_predictions),
    "precision": precision_score(y_test, lr_predictions),
    "recall": recall_score(y_test, lr_predictions),
    "f1": f1_score(y_test, lr_predictions),
}

print("\n--- Logistic Regression ---")
for k, v in lr_metrics.items():
    print(f"{k.capitalize()}: {v:.4f}")
print(classification_report(y_test, lr_predictions, target_names=["Legitimate", "Phishing"]))

lr_cm = confusion_matrix(y_test, lr_predictions)
plt.figure(figsize=(5, 4))
sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"])
plt.title("Logistic Regression — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("results/logistic_regression_confusion_matrix.png")
plt.close()


# ---------------------------------------------------------------------------
# 7. Select best model and save for deployment
# ---------------------------------------------------------------------------
best_model, best_name = (
    (rf_model, "random_forest") if rf_metrics["f1"] >= lr_metrics["f1"]
    else (lr_model, "logistic_regression")
)

print(f"\nBest performing model (by F1-score): {best_name}")

joblib.dump(best_model, "models/phishing_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print("\nSaved model and vectorizer to models/ directory.")
print("Ready for deployment — see app.py")
