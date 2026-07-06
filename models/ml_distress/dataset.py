"""
models/ml_distress/dataset.py

Loads the UCI Taiwanese Bankruptcy Prediction dataset (id=572) and extracts
the canonical 10-feature matrix defined in features.py, plus the binary
"Bankrupt?" label.

DATASET DETAILS (for the Methodology tab / interview prep):
  - Source: UCI Machine Learning Repository, "Taiwanese Bankruptcy
    Prediction" dataset (id 572). Originally sourced from the Taiwan
    Economic Journal, 1999-2009.
  - 6,819 companies, 95 financial ratio features, 1 binary label
    ("Bankrupt?": 1 = went bankrupt, 0 = did not).
  - Severe class imbalance: bankrupt firms are only ~3.2% of the dataset.
    This is handled at training time via class_weight="balanced" in the
    logistic regression, and is reported honestly in the evaluation
    metrics rather than papered over with plain accuracy (which would be
    misleadingly high even from a model that always predicts "not
    bankrupt").
  - LIMITATION: Taiwanese companies under Taiwanese accounting standards
    from over 15 years ago — not Indian companies, not Ind-AS/IFRS, and
    not current. The model captures general distress PATTERNS in these
    10 ratios, not a probability calibrated to real-world Indian default
    rates. This is stated explicitly in the app.

This module requires network access to UCI's servers and therefore must be
run on your own machine, not in the build sandbox.
"""

from __future__ import annotations
import pandas as pd

from .features import FEATURE_NAMES, UCI_COLUMN_MAP, features_from_uci_row


def load_uci_bankruptcy_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y):
      X - DataFrame with columns = FEATURE_NAMES (canonical, clean names)
      y - Series of 0/1 bankruptcy labels
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise ImportError(
            "ucimlrepo is required to fetch the dataset. Install with:\n"
            "    pip install ucimlrepo"
        ) from e

    dataset = fetch_ucirepo(id=572)
    raw_X = dataset.data.features   # 95 columns
    raw_y = dataset.data.targets    # "Bankrupt?" column

    missing_cols = [c for c in UCI_COLUMN_MAP.values() if c not in raw_X.columns]
    if missing_cols:
        raise ValueError(
            f"Expected UCI columns not found (UCI may have changed column naming): {missing_cols}\n"
            f"Available columns sample: {list(raw_X.columns)[:10]}"
        )

    X = pd.DataFrame(
        {name: raw_X[UCI_COLUMN_MAP[name]].astype(float) for name in FEATURE_NAMES}
    )
    y = raw_y.iloc[:, 0].astype(int)
    y.name = "bankrupt"

    return X, y


if __name__ == "__main__":
    X, y = load_uci_bankruptcy_dataset()
    print(f"Loaded {len(X)} companies, {X.shape[1]} canonical features.")
    print(f"Bankrupt class balance: {y.mean()*100:.2f}% positive ({y.sum()} of {len(y)})")
    print("\nFeature summary:")
    print(X.describe())
