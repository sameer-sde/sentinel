"""
Day 2 - Part 3: Export the trained model to ONNX format.

Gotcha: onnxmltools' XGBoost converter requires feature names matching
'f0, f1, f2, ...'. Our dataset has 'V1..V28, Time, Amount'. We retrain
with generic names before exporting. Same model architecture, identical
predictions - only column labels change.

Run: python 04_export_onnx.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort


# 1. Load data
print("Loading data...")
df = pd.read_csv("../data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

ORIGINAL_FEATURE_NAMES = X.columns.tolist()
print(f"  Original feature names: {ORIGINAL_FEATURE_NAMES[:5]}... ({len(ORIGINAL_FEATURE_NAMES)} total)")

X.columns = [f"f{i}" for i in range(X.shape[1])]
print(f"  Renamed to:             {list(X.columns[:5])}... ({len(X.columns)} total)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)


# 2. Retrain with the winning config
print("\nRetraining XGBoost (scale_pos_weight) with generic feature names...")
ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    eval_metric="logloss",
    scale_pos_weight=ratio,
    random_state=42,
)
model.fit(X_train, y_train)
print(f"  Trained. n_features={X.shape[1]}, scale_pos_weight={ratio:.0f}")


# 3. Convert to ONNX
print("\nConverting to ONNX...")
initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

onnx_path = "../models/fraud_model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

size_kb = os.path.getsize(onnx_path) / 1024
print(f"  Saved to {onnx_path}")
print(f"  ONNX file size: {size_kb:.1f} KB")


# 4. Verify Python vs ONNX predictions match
print("\nLoading ONNX model into onnxruntime for verification...")
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name

print("Running Python XGBoost predictions...")
py_proba = model.predict_proba(X_test.astype(np.float32))[:, 1]

print("Running ONNX predictions...")
onnx_input = X_test.astype(np.float32).values
onnx_output = session.run(None, {input_name: onnx_input})
onnx_proba = np.array([row[1] for row in onnx_output[1]], dtype=np.float32)


# 5. Compare
diff = np.abs(py_proba - onnx_proba)
max_diff = diff.max()
mean_diff = diff.mean()

print("\n" + "=" * 60)
print("PYTHON vs ONNX prediction comparison")
print("=" * 60)
print(f"  Max absolute difference:  {max_diff:.2e}")
print(f"  Mean absolute difference: {mean_diff:.2e}")
print(f"  Samples compared:         {len(py_proba):,}")

if max_diff < 1e-5:
    print("\n  [PASS] Predictions match. ONNX is safe to deploy.")
else:
    print("\n  [WARN] Predictions diverge. Investigate before deploying.")


# 6. Sample predictions
print(f"\nSample predictions (first 5 test rows):")
print(f"{'Index':>8} {'Python':>12} {'ONNX':>12} {'Diff':>12}")
for i in range(5):
    print(f"{i:>8} {py_proba[i]:>12.6f} {onnx_proba[i]:>12.6f} {diff[i]:>12.2e}")


# 7. Save feature mapping for the Go server
mapping = {
    "feature_order": ORIGINAL_FEATURE_NAMES,
    "generic_names": [f"f{i}" for i in range(len(ORIGINAL_FEATURE_NAMES))],
    "n_features": len(ORIGINAL_FEATURE_NAMES),
}
mapping_path = "../models/feature_mapping.json"
with open(mapping_path, "w") as f:
    json.dump(mapping, f, indent=2)
print(f"\nSaved feature mapping to {mapping_path}")

print("\nDone. ONNX model ready for the Go serving layer.")
