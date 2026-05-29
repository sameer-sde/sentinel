"""
Day 0: Initial data exploration.

Goal: understand the fraud dataset before touching any model.
- How many rows, columns?
- What does a row look like?
- How severe is the class imbalance?
- What are the feature ranges?

Run: python 00_explore.py
"""

import pandas as pd
import numpy as np

# 1. Load it. low_memory=False is safer for files with mixed-type columns.
print("Loading dataset...")
df = pd.read_csv("../data/creditcard.csv")

# 2. Basic shape
print(f"\n=== Shape ===")
print(f"Rows:    {len(df):,}")
print(f"Columns: {df.shape[1]}")

# 3. What do the columns look like?
print(f"\n=== Columns ===")
print(df.columns.tolist())

# 4. First few rows (transposed so it fits the screen)
print(f"\n=== First 3 rows (transposed) ===")
print(df.head(3).T)

# 5. The big one: class balance
print(f"\n=== Class distribution ===")
class_counts = df["Class"].value_counts()
print(class_counts)
print(f"\nLegit:  {class_counts[0]:,} ({class_counts[0] / len(df) * 100:.4f}%)")
print(f"Fraud:  {class_counts[1]:,} ({class_counts[1] / len(df) * 100:.4f}%)")
print(f"Ratio:  1 fraud per {class_counts[0] // class_counts[1]:,} legit transactions")

# 6. Are there missing values? (a real-world dataset can be a mess; this one is clean, but check)
print(f"\n=== Missing values per column ===")
missing = df.isna().sum()
if missing.sum() == 0:
    print("None. Clean dataset.")
else:
    print(missing[missing > 0])

# 7. Range of the two human-readable features: Time and Amount
print(f"\n=== Time range (seconds since first transaction) ===")
print(f"min: {df['Time'].min():.0f}")
print(f"max: {df['Time'].max():.0f}")
print(f"span: {df['Time'].max() / 3600:.1f} hours ({df['Time'].max() / 86400:.1f} days)")

print(f"\n=== Amount stats ===")
print(df["Amount"].describe())

# 8. Compare fraud vs legit amounts — do they differ?
print(f"\n=== Amount: fraud vs legit ===")
print("Legit transactions:")
print(df.loc[df["Class"] == 0, "Amount"].describe())
print("\nFraud transactions:")
print(df.loc[df["Class"] == 1, "Amount"].describe())

print("\nDone.")
