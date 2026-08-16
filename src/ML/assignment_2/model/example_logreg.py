"""
WORKED EXAMPLE — Logistic Regression, end to end.

This is a complete, runnable reference for ONE model. Use it as a template to
fill in train_models.py for all 5 models. Every step below maps to a TODO in
the skeleton.

Run:  python model/example_logreg.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

# ---- config ----
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "bank-full.csv"
POS_LABEL = "yes"
DROP_DURATION = True
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ---- 1. load ----
df = pd.read_csv(DATA_PATH, sep=";")          # semicolon-separated UCI file

# ---- 2. features / target ----
y = (df["y"] == POS_LABEL).astype(int)        # encode target to 1/0
X = df.drop(columns=["y"])
if DROP_DURATION:
    X = X.drop(columns=["duration"])          # remove leakage feature

# ---- 3. preprocessing (bundled INTO the pipeline, so it saves with the model) ----
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# ---- 4. split (stratified because the target is imbalanced) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ---- 5. build + fit the pipeline ----
pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
pipe.fit(X_train, y_train)

# ---- 6. evaluate: all 6 required metrics on the test set ----
pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]      # P(class=1) for AUC
metrics = {
    "accuracy":  round(accuracy_score(y_test, pred), 4),
    "auc":       round(roc_auc_score(y_test, proba), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall":    round(recall_score(y_test, pred), 4),
    "f1":        round(f1_score(y_test, pred), 4),
    "mcc":       round(matthews_corrcoef(y_test, pred), 4),
}

# ---- 7. save the fitted pipeline (raw CSV -> prediction, self-contained) ----
joblib.dump(pipe, HERE / "logistic_regression.joblib")

print("Train rows:", len(X_train), "| Test rows:", len(X_test))
print("Numeric cols:", num_cols)
print("Categorical cols:", cat_cols)
print("\nLogistic Regression metrics:")
print(json.dumps(metrics, indent=2))
print("\nSaved -> model/logistic_regression.joblib")
