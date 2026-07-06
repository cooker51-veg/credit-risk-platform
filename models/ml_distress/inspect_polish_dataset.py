"""
models/ml_distress/inspect_polish_dataset.py

DIAGNOSTIC SCRIPT — run this once, send Claude the output, before any
feature-mapping code is written for the Polish Companies Bankruptcy
dataset (UCI id=365). This avoids repeating the exact mistake found in the
Taiwanese dataset: guessing at column meaning from memory instead of
verifying it against the real, published attribute descriptions.
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

dataset = fetch_ucirepo(id=365)

print("=" * 80)
print("VARIABLE METADATA (name + description, as published by UCI)")
print("=" * 80)
print(dataset.variables)

print("\n" + "=" * 80)
print("FEATURE SUMMARY STATISTICS (first 15 columns)")
print("=" * 80)
X = dataset.data.features
print(X.iloc[:, :15].describe())

print(f"\nTotal shape: {X.shape}")
print(f"Target distribution:\n{dataset.data.targets.iloc[:, 0].value_counts()}")
