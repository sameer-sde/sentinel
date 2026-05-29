"""
Day 2 - Part 2: Cost-based threshold optimization.

We've established that recall vs precision is a trade-off. Now we
formalize the trade-off with a business cost model and pick the
threshold that minimizes total cost.

Cost assumptions (tuneable):
  - Missed fraud (FN):  ₹500 per case (lost transaction + chargeback overhead)
  - False alarm (FP):   ₹50 per case  (support cost + customer friction)

Cost ratio: 10:1. Adjust to your business reality.

Run: python 03_threshold.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import xgboost as xgb


# -------------------- 1. Load winner, get predictions --------------------
print("Loading data + improved model...")
df = pd.read_csv("../data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

model = xgb.XGBClassifier()
model.load_model("../models/improved.json")  # the scale_pos_weight winner

y_proba = model.predict_proba(X_test)[:, 1]


# -------------------- 2. Cost model --------------------
COST_FN = 500   # cost of missing one fraud (₹)
COST_FP = 50    # cost of one false alarm (₹)

def total_cost(threshold, y_true, y_proba):
    """Total business cost at this threshold."""
    pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return fn * COST_FN + fp * COST_FP, fn, fp, tp


# -------------------- 3. Sweep thresholds densely --------------------
print("\nScanning 100 threshold values from 0.01 to 0.99...")
print(f"\n{'Threshold':>10} {'Cost (₹)':>12} {'Missed':>8} {'Alarms':>8} {'Caught':>8} {'Precision':>11} {'Recall':>9}")
print("-" * 75)

thresholds = np.linspace(0.01, 0.99, 99)
results = []

# Sample print every ~10th threshold for readability
print_every = 10
for i, t in enumerate(thresholds):
    cost, fn, fp, tp = total_cost(t, y_test, y_proba)
    pred = (y_proba >= t).astype(int)
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred)
    results.append({"threshold": t, "cost": cost, "fn": fn, "fp": fp, "tp": tp,
                    "precision": p, "recall": r})
    if i % print_every == 0:
        print(f"{t:>10.2f} {cost:>12,} {fn:>8} {fp:>8} {tp:>8} {p*100:>10.2f}% {r*100:>8.2f}%")


# -------------------- 4. The optimal threshold --------------------
results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df["cost"].idxmin()]

print("\n" + "=" * 60)
print("OPTIMAL THRESHOLD (cost-minimizing)")
print("=" * 60)
print(f"Threshold:        {best_row['threshold']:.3f}")
print(f"Total cost:       ₹{best_row['cost']:,.0f}")
print(f"Missed frauds:    {int(best_row['fn'])}/98  (₹{int(best_row['fn']) * COST_FN:,})")
print(f"False alarms:     {int(best_row['fp'])}     (₹{int(best_row['fp']) * COST_FP:,})")
print(f"Caught frauds:    {int(best_row['tp'])}/98")
print(f"Precision:        {best_row['precision']*100:.2f}%")
print(f"Recall:           {best_row['recall']*100:.2f}%")


# -------------------- 5. Compare to default 0.5 --------------------
default = results_df.iloc[49]  # threshold ≈ 0.50
savings = default["cost"] - best_row["cost"]

print("\n" + "=" * 60)
print("WHAT WE GAINED BY TUNING")
print("=" * 60)
print(f"Default threshold 0.5 cost:    ₹{default['cost']:,.0f}")
print(f"Optimal threshold {best_row['threshold']:.2f} cost:    ₹{best_row['cost']:,.0f}")
print(f"Savings:                       ₹{savings:,.0f}")
print(f"Extrapolated to 1M txns/day:   ₹{savings * (1_000_000 / len(y_test)):,.0f}/day saved")

# Save the optimal threshold for the serving layer to use
import json, os
os.makedirs("../models", exist_ok=True)
config = {
    "optimal_threshold": float(best_row['threshold']),
    "cost_fn": COST_FN,
    "cost_fp": COST_FP,
    "expected_recall": float(best_row['recall']),
    "expected_precision": float(best_row['precision']),
}
with open("../models/threshold_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"Saved threshold config to ../models/threshold_config.json")

print("\nDone.")
