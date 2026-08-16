"""
Train & evaluate 5 classification models on the Bank Marketing dataset.

Fill in every TODO. When done, running this script should:
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
    """Read the semicolon-separated CSV and return a DataFrame.

    Hint: pd.read_csv(..., sep=";")
    """
    return pd.read_csv(path, sep=";")


# ----------------------------------------------------------------------
# 2. Split features / target
# ----------------------------------------------------------------------
def split_X_y(df):
    """Return (X, y).

    - y: the `y` column, encoded to 0/1 (1 == POS_LABEL)
    - X: all other columns; drop `duration` if DROP_DURATION
    Hint: (df["y"] == POS_LABEL).astype(int)
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
    """Return a ColumnTransformer that:
        - StandardScaler on numeric columns
        - OneHotEncoder(handle_unknown="ignore") on categorical columns
    Hints:
        num_cols = X.select_dtypes(include="number").columns
        cat_cols = X.select_dtypes(exclude="number").columns
    """
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
    """Return an ordered dict {display_name: estimator} for the 5 models.

    Suggested (tune later):
        Logistic Regression  -> LogisticRegression(max_iter=..., class_weight="balanced")
        Decision Tree        -> DecisionTreeClassifier(class_weight="balanced", random_state=...)
        kNN                  -> KNeighborsClassifier(n_neighbors=...)
        Naive Bayes          -> GaussianNB()
        Random Forest        -> RandomForestClassifier(n_estimators=..., class_weight="balanced", ...)
    Note: kNN and GaussianNB do NOT accept class_weight.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        # TODO: add the other 4 models here, e.g.
        # "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        # "kNN": KNeighborsClassifier(n_neighbors=15),
        # "Naive Bayes": GaussianNB(),
        # "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
        #                                         random_state=RANDOM_STATE, n_jobs=-1),
    }


# ----------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------
def evaluate(pipe, X_test, y_test):
    """Return a dict with keys: accuracy, auc, precision, recall, f1, mcc.

    - Use pipe.predict(X_test) for the class predictions.
    - Use pipe.predict_proba(X_test)[:, 1] for AUC.
    - precision/recall/f1 for the positive class (pos_label=1 -> default binary).
    - mcc via matthews_corrcoef.
    Round to 4 decimals for readability.
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

    preprocessor = build_preprocessor(X_train)
    results = {}

    for name, clf in get_models().items():
        pipe = Pipeline([("pre", clone(preprocessor)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        slug = name.lower().replace(" ", "_")
        joblib.dump(pipe, HERE / f"{slug}.joblib")
        results[name] = evaluate(pipe, X_test, y_test)
        print(f"  {name:<22} {results[name]}")

    with open(HERE / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved models + metrics.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
