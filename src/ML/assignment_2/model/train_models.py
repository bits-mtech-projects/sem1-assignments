"""
Train & evaluate 5 classification models on the Bank Marketing dataset.


  1. load data/bank-full.csv
  2. drop the leaky `duration` column
  3. split into train/test (stratified)
  4. build one sklearn Pipeline per model (preprocessing + classifier)
  5. fit each, compute 6 metrics on the test set
  6. save each fitted pipeline to model/<name>.joblib
  7. write all metrics to model/metrics.json

Run:  python model/train_models.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../model
DATA_PATH = HERE.parent / "data" / "bank-full.csv"
POS_LABEL = "yes"          # positive class of the target `y`
DROP_DURATION = True       # leakage — see README
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
def load_data(path):
    """Read the semicolon-separated Bank Marketing CSV into a DataFrame."""
    return pd.read_csv(path, sep=";")


# ----------------------------------------------------------------------
# 2. Split features / target
# ----------------------------------------------------------------------
def split_X_y(df):
    """Split into features X and binary target y (1 == subscribed).

    The leaky `duration` column is removed when DROP_DURATION is set.
    """
    y = (df["y"] == POS_LABEL).astype(int)
    X = df.drop(columns=["y"])
    if DROP_DURATION:
        X = X.drop(columns=["duration"])
    return X, y


# ----------------------------------------------------------------------
# 3. Preprocessing
# ----------------------------------------------------------------------
def build_preprocessor(X):
    """Standard-scale the numeric columns and one-hot encode the categorical ones."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


# ----------------------------------------------------------------------
# 4. Models
# ----------------------------------------------------------------------
def get_models():
    """Return the five classifiers keyed by display name.

    class_weight="balanced" is applied to the models that support it
    (Logistic Regression, Decision Tree, Random Forest) to offset the class
    imbalance; kNN and Gaussian Naive Bayes do not accept that argument.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=150,max_depth=20, class_weight="balanced",
                                                 random_state=RANDOM_STATE, n_jobs=-1),
    }


# ----------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------
def evaluate(pipe, X_test, y_test):
    """Compute the six evaluation metrics for a fitted pipeline on the test set.

    Precision, recall and F1 are for the positive class; AUC uses predicted
    probabilities. Values are rounded to 4 decimals.
    """
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, pred), 4),
        "auc":       round(roc_auc_score(y_test, proba), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall":    round(recall_score(y_test, pred), 4),
        "f1":        round(f1_score(y_test, pred), 4),
        "mcc":       round(matthews_corrcoef(y_test, pred), 4),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Export the held-out test set for the Streamlit app.
    # Same split as evaluation -> the app's metrics will match metrics.json / README.
    # y is written back as the original "yes"/"no" labels for readability; the app re-encodes.
    test_df = X_test.copy()
    test_df["y"] = y_test.map({1: "yes", 0: "no"})
    test_df.to_csv(HERE.parent / "test_data.csv", index=False)
    print(f"Exported test_data.csv ({len(test_df)} rows)")

    preprocessor = build_preprocessor(X_train)
    results = {}

    for name, clf in get_models().items():
        pipe = Pipeline([("pre", clone(preprocessor)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        slug = name.lower().replace(" ", "_")
        joblib.dump(pipe, HERE / f"{slug}.joblib", compress=3)
        results[name] = evaluate(pipe, X_test, y_test)
        print(f"  {name:<22} {results[name]}")

    with open(HERE / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved models + metrics.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
