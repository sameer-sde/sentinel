"""
Day 3: pick two specific test rows (one fraud, one legit), run them
through the Python ONNX session, and print everything Go will need to
verify itself against.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import onnxruntime as ort

# Same split as everywhere else.
df = pd.read_csv("../data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]
X.columns = [f"f{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Find one of each class in the test set
fraud_idx = y_test[y_test == 1].index[0]      # first known fraud
legit_idx = y_test[y_test == 0].index[0]      # first known legit

fraud_row = X_test.loc[fraud_idx].values.astype(np.float32)
legit_row = X_test.loc[legit_idx].values.astype(np.float32)

# Run them through ONNX
session = ort.InferenceSession("../models/fraud_model.onnx")
input_name = session.get_inputs()[0].name

# We need to feed a batch, so reshape to (1, 30)
fraud_input = fraud_row.reshape(1, -1)
legit_input = legit_row.reshape(1, -1)

fraud_out = session.run(None, {input_name: fraud_input})
legit_out = session.run(None, {input_name: legit_input})

# Extract fraud probabilities
fraud_proba = fraud_out[1][0][1]
legit_proba = legit_out[1][0][1]

print("=" * 60)
print("KNOWN FRAUD transaction")
print("=" * 60)
print(f"Actual label:        1 (fraud)")
print(f"Model probability:   {fraud_proba:.6f}")
print(f"Predicted class:     {int(fraud_out[0][0])}")
print(f"\nFeature values (Go needs these — copy into main.go):")
print(", ".join(f"{v:.6f}" for v in fraud_row))

print("\n" + "=" * 60)
print("KNOWN LEGIT transaction")
print("=" * 60)
print(f"Actual label:        0 (legit)")
print(f"Model probability:   {legit_proba:.6f}")
print(f"Predicted class:     {int(legit_out[0][0])}")
print(f"\nFeature values (Go needs these — copy into main.go):")
print(", ".join(f"{v:.6f}" for v in legit_row))
