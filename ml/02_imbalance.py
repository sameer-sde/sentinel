"""
Day 2 - Part 1: Handle the class imbalance.

Two approaches compared head-to-head:
  A) scale_pos_weight (XGBoost built-in)
  B) SMOTE (synthetic minority oversampling)

Baseline (no handling) from Day 1: PR-AUC 0.84, recall 77.55%.
Goal: beat it. Measurably.

Run: python 02_imbalance.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb


# -------------------- 1. Load + split --------------------
print("Loading data...")
df = pd.read_csv("../data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)


def evaluate(name, model, X_te, y_te):
    """Print a compact metrics line for one model."""
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    prec = precision_score(y_te, pred, zero_division=0)
    rec = recall_score(y_te, pred)
    f1 = f1_score(y_te, pred)
    roc_auc = roc_auc_score(y_te, proba)
    pr_auc = average_precision_score(y_te, proba)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    print(f"\n--- {name} ---")
    print(f"  Precision: {prec*100:>6.2f}%    Recall: {rec*100:>6.2f}%    F1: {f1*100:>6.2f}%")
    print(f"  ROC-AUC:   {roc_auc:.4f}    PR-AUC: {pr_auc:.4f}")
    print(f"  Caught: {tp}/{tp+fn} frauds    False alarms: {fp}")

    return {"name": name, "pr_auc": pr_auc, "recall": rec, "precision": prec,
            "model": model, "proba": proba}


# -------------------- 2. Approach A: scale_pos_weight --------------------
# Ratio: (number of legit) / (number of fraud). XGBoost uses this to up-weight
# fraud examples during training. Equivalent to oversampling without the cost.
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass ratio (legit:fraud) = {ratio:.0f}:1")
print("Training XGBoost with scale_pos_weight...")

model_spw = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    eval_metric="logloss",
    scale_pos_weight=ratio,   # the key line
)
model_spw.fit(X_train, y_train)
results_spw = evaluate("scale_pos_weight", model_spw, X_test, y_test)


# -------------------- 3. Approach B: SMOTE --------------------
# SMOTE works on the TRAINING set only — we never resample test set, because
# we need to evaluate on the real-world distribution.
print("\nApplying SMOTE to training set...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"  Before SMOTE: {len(X_train):,} rows, {y_train.sum()} frauds")
print(f"  After SMOTE:  {len(X_train_smote):,} rows, {y_train_smote.sum()} frauds")

print("Training XGBoost on SMOTE-resampled data...")
model_smote = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    eval_metric="logloss",
)
model_smote.fit(X_train_smote, y_train_smote)
results_smote = evaluate("SMOTE", model_smote, X_test, y_test)


# -------------------- 4. Approach C: baseline reload (for comparison) --------------------
# Reload yesterday's baseline so the comparison is honest.
print("\nReloading yesterday's baseline for comparison...")
baseline = xgb.XGBClassifier()
baseline.load_model("../models/baseline.json")
results_baseline = evaluate("baseline (no handling)", baseline, X_test, y_test)


# -------------------- 5. Compare --------------------
print("\n" + "=" * 60)
print("SIDE-BY-SIDE")
print("=" * 60)
print(f"{'Approach':<25} {'PR-AUC':>10} {'Recall':>10} {'Precision':>12}")
for r in [results_baseline, results_spw, results_smote]:
    print(f"{r['name']:<25} {r['pr_auc']:>10.4f} {r['recall']*100:>9.2f}% {r['precision']*100:>11.2f}%")


# -------------------- 6. Pick the winner --------------------
winner = max(
    [results_spw, results_smote, results_baseline],
    key=lambda r: r["pr_auc"]
)
print(f"\nWinner by PR-AUC: {winner['name']} ({winner['pr_auc']:.4f})")

# Save the winner as the "improved" model
os.makedirs("../models", exist_ok=True)
winner["model"].save_model("../models/improved.json")
print(f"Saved {winner['name']} model to ../models/improved.json")

print("\nDone.")
