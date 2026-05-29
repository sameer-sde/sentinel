"""
Day 1: Baseline XGBoost model with proper imbalanced-data evaluation.

This is a 'naive baseline' — no resampling, no class weighting, no
hyperparameter tuning. We're establishing what we get with defaults so
we know how much our later work actually helps.

Run: python 01_baseline.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
import xgboost as xgb


# -------------------- 1. Load + split --------------------
print("Loading data...")
df = pd.read_csv("../data/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# stratify=y ensures both train and test get the same fraud proportion.
# Without it, with only 492 frauds, a random split could give you almost
# none in test — which would make evaluation useless.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"Train: {len(X_train):,} rows, {y_train.sum()} frauds ({y_train.mean()*100:.3f}%)")
print(f"Test:  {len(X_test):,} rows, {y_test.sum()} frauds ({y_test.mean()*100:.3f}%)")


# -------------------- 2. Train baseline XGBoost --------------------
print("\nTraining baseline XGBoost (default hyperparameters)...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    # NO class weighting — this is the *naive* baseline.
)
model.fit(X_train, y_train)


# -------------------- 3. Predict --------------------
# .predict() uses default threshold = 0.5
y_pred = model.predict(X_test)
# .predict_proba() gives probability, which lets us vary the threshold.
y_proba = model.predict_proba(X_test)[:, 1]  # column 1 = probability of fraud


# -------------------- 4. The metrics dump --------------------
print("\n" + "=" * 60)
print("METRICS @ default threshold 0.5")
print("=" * 60)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:       {acc*100:.4f}%   <-- DON'T be fooled by this")
print(f"Precision:      {prec*100:.2f}%   (when we say fraud, how often correct)")
print(f"Recall:         {rec*100:.2f}%   (of all real fraud, how much we caught)")
print(f"F1 score:       {f1*100:.2f}%")
print(f"ROC-AUC:        {roc_auc:.4f}")
print(f"PR-AUC:         {pr_auc:.4f}   <-- the metric that matters here")

print(f"\nConfusion matrix:")
print(f"                  Predicted Legit   Predicted Fraud")
print(f"Actual Legit:       {cm[0,0]:>8,}        {cm[0,1]:>8,}")
print(f"Actual Fraud:       {cm[1,0]:>8,}        {cm[1,1]:>8,}")

tn, fp, fn, tp = cm.ravel()
print(f"\nIn plain terms:")
print(f"  TP (caught fraud):       {tp}")
print(f"  FN (missed fraud):       {fn}  <-- money walked away")
print(f"  FP (false alarms):       {fp}  <-- innocent customers blocked")
print(f"  TN (correctly cleared):  {tn:,}")


# -------------------- 5. Threshold sweep --------------------
# Show what happens at different decision thresholds.
print("\n" + "=" * 60)
print("THRESHOLD SWEEP — precision/recall trade-off")
print("=" * 60)
print(f"{'Threshold':>10} {'Precision':>12} {'Recall':>10} {'Fraud caught':>14} {'False alarms':>14}")

for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    pred_at_t = (y_proba >= t).astype(int)
    p = precision_score(y_test, pred_at_t, zero_division=0)
    r = recall_score(y_test, pred_at_t)
    caught = ((pred_at_t == 1) & (y_test == 1)).sum()
    alarms = ((pred_at_t == 1) & (y_test == 0)).sum()
    print(f"{t:>10.2f} {p*100:>11.2f}% {r*100:>9.2f}% {caught:>14} {alarms:>14}")


# -------------------- 6. Save model --------------------
# Save the baseline model so we can compare against it later
# (and so Day 2's improved model has something to beat).
os.makedirs("../models", exist_ok=True)
model.save_model("../models/baseline.json")
print("\nSaved baseline model to ../models/baseline.json")

print("\nDone. Look at the threshold sweep — that's the lever the business pulls.")
