"""
Bank Marketing — Term Deposit Subscription Classifier (Streamlit app).

Loads the 5 pre-trained model pipelines and lets the user:
  - upload a test CSV (or use the bundled test_data.csv),
  - pick a model,
  - see the 6 evaluation metrics + a confusion matrix / classification report,
  - compare all 5 models on the same data.

Run locally:  streamlit run app.py

>>> PERSONALIZE the marked spots (title / intro / your name) — the assignment
    penalizes un-customized template apps.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

# ----------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
DEFAULT_TEST = HERE / "test_data.csv"
TARGET = "y"
POS_LABEL = "yes"
CLASS_NAMES = ["no", "yes"]

# display name -> saved joblib slug
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
}


# ----------------------------------------------------------------------
# Loading (cached so models load once per session)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Load every saved pipeline into {display_name: fitted_pipeline}."""
    loaded = {}
    for name, slug in MODELS.items():
        path = MODEL_DIR / f"{slug}.joblib"
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded


@st.cache_data(show_spinner=False)
def load_default_test():
    return pd.read_csv(DEFAULT_TEST)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def split_features_target(df):
    """Return (X, y_true) where y_true is 1/0 (or None if no target column)."""
    if TARGET in df.columns:
        y_true = (df[TARGET] == POS_LABEL).astype(int)
        X = df.drop(columns=[TARGET])
    else:
        y_true, X = None, df
    return X, y_true


def compute_metrics(pipe, X, y_true):
    """Return the 6 required metrics as a dict (test-set style evaluation)."""
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, proba),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.set_page_config(page_title="Bank Marketing Classifier", page_icon="🏦", layout="wide")


st.title("🏦 Bank Marketing — Term Deposit Subscription Classifier")
st.caption("Predicting whether a client subscribes to a term deposit — "
           "comparing 5 classifiers on held-out test data.  \n"
           "_By: Mohammad Uzair, BITS ID: 2025AC05743")

models = load_models()
if not models:
    st.error("No model files found in model/. Run model/train_models.py first.")
    st.stop()

# ---- Sidebar controls ----
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
selected_name = st.sidebar.selectbox("Choose a model", list(models.keys()), index=len(models) - 1)
st.sidebar.caption("No file? The app uses the bundled `test_data.csv`.")

# ---- Load the data to evaluate on ----
if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.success(f"Using uploaded file — {len(data)} rows.")
else:
    data = load_default_test()
    st.info(f"Using bundled test_data.csv — {len(data)} rows.")

X, y_true = split_features_target(data)

with st.expander("Preview data", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)

if y_true is None:
    st.warning(f"No '{TARGET}' column found, so metrics can't be computed. "
               "Showing predictions only.")
    preds = models[selected_name].predict(X)
    st.dataframe(pd.DataFrame({"prediction": np.where(preds == 1, "yes", "no")}))
    st.stop()

# ----------------------------------------------------------------------
# Selected-model results
# ----------------------------------------------------------------------
st.subheader(f"Results — {selected_name}")

pipe = models[selected_name]
metrics = compute_metrics(pipe, X, y_true)

cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.3f}")

left, right = st.columns(2)

with left:
    st.markdown("**Confusion matrix**")
    pred = pipe.predict(X)
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(y_true, pred, target_names=CLASS_NAMES, zero_division=0)
    st.code(report)

# ----------------------------------------------------------------------
# All-models comparison on the SAME data
# ----------------------------------------------------------------------
st.subheader("All models on this data")

rows = []
for name, model in models.items():
    m = compute_metrics(model, X, y_true)
    rows.append({"Model": name, **{k: round(v, 4) for k, v in m.items()}})
comparison = pd.DataFrame(rows).set_index("Model")

st.dataframe(
    comparison.style.highlight_max(axis=0, color="#c6efce").format("{:.4f}"),
    use_container_width=True,
)
st.caption("Green = best value per metric.  Metrics computed live on the data above.")
