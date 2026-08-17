# Bank Marketing — Term Deposit Subscription Classifier

Machine Learning Assignment 2 — five classification models trained on the UCI Bank
Marketing dataset, compared on held-out test data, and served through an interactive
Streamlit web app.

## a. Problem statement
A Portuguese bank ran a direct (phone-call) marketing campaign to sell term deposits.
The goal is to predict, **before a client is contacted**, whether they will subscribe
to a term deposit (`y` = yes/no), using only demographic, financial, and previous-campaign
attributes. This is a **binary, class-imbalanced** classification problem (only ~11.7% of
clients subscribe), so the bank can target likely subscribers and reduce wasted calls.

## b. Dataset description
- **Source:** UCI Machine Learning Repository — *Bank Marketing* (Moro, Cortez & Rita, 2014).
  https://archive.ics.uci.edu/dataset/222/bank+marketing  (file: `bank-full.csv`)
- **Instances:** 45,211
- **Features used:** 15 (the raw dataset has 16 inputs; the `duration` column is **dropped** — see note below)
- **Target:** `y` — whether the client subscribed to a term deposit (`yes`/`no`); ~11.7% positive (imbalanced)
- **Feature types:** 6 numeric (`age`, `balance`, `day`, `campaign`, `pdays`, `previous`) and
  9 categorical (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`)
- **Train/test split:** 80/20 stratified (36,168 train / 9,043 test rows), `random_state=42`

> **Why `duration` is dropped:** `duration` (last-call length in seconds) is only known *after*
> a call ends, and it almost perfectly reveals the outcome — a textbook data-leakage feature.
> The UCI authors explicitly recommend excluding it for a realistic model. Dropping it lowered
> AUC by ~0.14 in our tests, confirming it was leaking the answer. 15 features remain (≥ 12 required).

## c. GitHub Repository Link
https://github.com/bits-mtech-projects/sem1-assignments — project path: `src/ML/assignment_2/`

## d. Models used

All five models are built as scikit-learn `Pipeline`s (`StandardScaler` on numeric +
`OneHotEncoder` on categorical features → classifier), trained on the training split, and
evaluated on the held-out test set. `class_weight="balanced"` is used where supported
(Logistic Regression, Decision Tree, Random Forest) to counter the class imbalance.

### Comparison table (test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.3732 | 0.2853 |
| Decision Tree | 0.8429 | 0.6081 | 0.3187 | 0.3015 | 0.3099 | 0.2214 |
| kNN | 0.8921 | 0.7349 | 0.6475 | 0.1701 | 0.2695 | 0.2939 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Random Forest (Ensemble) | 0.8661 | **0.7984** | 0.4357 | 0.4896 | **0.4611** | **0.3858** |

_Precision / Recall / F1 are reported for the positive class (`yes`)._

### Observations

| ML Model Name | Observation about model performance                                                                                                                                                                                                                                           |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression | Lowest accuracy (0.75) but strong recall (0.62): with balanced class weights it flags most true subscribers, at the cost of many false positives (precision 0.27). Good AUC (0.77) shows solid ranking ability. Useful when catching subscribers matters more than precision. |
| Decision Tree | Weakest overall, lowest AUC (0.61, barely above random ranking) and lowest MCC (0.22). Its 0.84 accuracy is misleading: a single tree mostly predicts the majority class and gives poor probability estimates, so it generalises worst.                                       |
| kNN | Highest raw accuracy (0.89) but that is deceptive on an imbalanced dataset — recall is only 0.17, so it misses ~83% of real subscribers by defaulting to "no". High precision (0.65) when it does predict "yes". No class-weight support hurts it here.                       |
| Naive Bayes | Surprisingly competitive and well-balanced (F1 0.41, MCC 0.33 — 2nd best). Fast and simple; even though its feature-independence assumption is violated, it handles the imbalance better than the tree/kNN.                                                                   |
| Random Forest (Ensemble) | **Best model** — highest AUC (0.798), F1 (0.461) and MCC (0.386), with balanced precision/recall. Averaging 150 depth-capped trees controls overfitting and produces the most reliable, well-ranked predictions.                                                              |
| **Overall Winner for your dataset?** | **Random Forest.** On this imbalanced problem, accuracy is misleading (kNN "wins" accuracy while being nearly useless at finding subscribers). Judged by the metrics that matter for imbalance — AUC, F1 and MCC — Random Forest is clearly best.                             |

## Live Streamlit App
<!-- TODO: paste your deployed Streamlit Community Cloud URL here after deployment -->

## Streamlit app features
- **Upload test CSV** (falls back to the bundled `test_data.csv`)
- **Model selection dropdown** (all 5 models)
- **Evaluation metrics** for the selected model (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Confusion matrix** + classification report
- **Live 5-model comparison table** on the uploaded data

## Project structure
```
src/ML/assignment_2/
├── app.py                 # Streamlit app
├── requirements.txt       # pinned dependencies
├── README.md              # this file
├── test_data.csv          # held-out test set (upload target for the app)
├── data/                  # raw dataset (bank-full.csv)
└── model/
    ├── train_models.py    # trains, evaluates, saves all 5 pipelines + metrics.json
    ├── *.joblib           # saved model pipelines
    └── metrics.json       # comparison-table numbers
```

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to reproduce the models
```bash
python model/train_models.py   # regenerates the .joblib files, metrics.json, and test_data.csv
```
