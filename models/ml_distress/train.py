"""
models/ml_distress/train.py

Trains the distress prediction model on the UCI Taiwanese Bankruptcy
dataset using the canonical 10-feature set.

Model choice: Logistic Regression, NOT a black box.
  - Coefficients are directly interpretable (sign + magnitude per feature,
    on standardized scale).
  - class_weight="balanced" compensates for the ~3.2% positive class rate
    without arbitrarily oversampling/undersampling.
  - SHAP (via LinearExplainer, exact for logistic regression) gives
    per-prediction feature attribution — the "why was THIS company
    flagged" answer required by the master prompt.

A Gradient Boosting model is also trained as a comparison point (higher
capacity, still tree-based and SHAP-explainable via TreeExplainer), and
both models' test-set metrics are printed side by side so you can discuss
the accuracy/explainability tradeoff in interviews.

Run this on your own machine (needs internet access to UCI's servers):
    pip install ucimlrepo scikit-learn shap joblib
    python models/ml_distress/train.py
"""

from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .dataset import load_uci_bankruptcy_dataset
from .features import FEATURE_NAMES

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# The UCI Taiwanese Bankruptcy dataset is known to contain a small number of
# extreme outliers in a few ratio columns (notably Current Ratio / Quick
# Ratio), almost certainly data errors from the original source rather than
# real companies with a current ratio in the thousands. Left unwinsorized,
# these single rows dominate both the scaler's variance estimate and SHAP's
# mean |contribution|, producing nonsensical numbers (a SHAP value in the
# millions) despite a tiny, sane model coefficient. Clipping at the 1st/99th
# percentile (computed on TRAIN only, to avoid leakage) fixes this. The
# clip bounds are saved as an artifact so predict.py applies the identical
# transform to live company data.
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99


def train_and_evaluate():
    print("Loading UCI Taiwanese Bankruptcy dataset...")
    X, y = load_uci_bankruptcy_dataset()
    print(f"  {len(X)} companies | {y.mean()*100:.2f}% bankrupt (severe class imbalance)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Winsorize outliers (fit bounds on train only) ---
    lower_bounds = X_train.quantile(WINSOR_LOWER_Q)
    upper_bounds = X_train.quantile(WINSOR_UPPER_Q)
    n_clipped = ((X_train < lower_bounds) | (X_train > upper_bounds)).sum().sum()
    print(f"\nWinsorizing at [{WINSOR_LOWER_Q}, {WINSOR_UPPER_Q}] percentiles "
          f"({n_clipped} training values clipped across all features)...")

    X_train = X_train.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    X_test = X_test.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Primary model: Logistic Regression (fully explainable) ---
    print("\nTraining Logistic Regression (class_weight='balanced')...")
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)

    logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_preds = logreg.predict(X_test_scaled)
    _report("Logistic Regression", y_test, logreg_preds, logreg_probs)

    print("\nLogistic Regression coefficients (standardized scale):")
    for name, coef in sorted(zip(FEATURE_NAMES, logreg.coef_[0]), key=lambda t: -abs(t[1])):
        direction = "higher distress risk" if coef > 0 else "lower distress risk"
        print(f"  {name:40s} {coef:+.4f}  ({direction})")

    # --- Secondary comparison model: Gradient Boosting ---
    print("\nTraining Gradient Boosting (comparison model)...")
    gb = GradientBoostingClassifier(random_state=42)
    # GB doesn't take class_weight directly; approximate via sample_weight.
    sample_weight = np.where(y_train == 1, (y_train == 0).sum() / (y_train == 1).sum(), 1.0)
    gb.fit(X_train, y_train, sample_weight=sample_weight)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    gb_preds = gb.predict(X_test)
    _report("Gradient Boosting", y_test, gb_preds, gb_probs)

    # --- SHAP explainability ---
    print("\nComputing SHAP values (Logistic Regression, LinearExplainer)...")
    explainer = shap.LinearExplainer(logreg, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    print("Mean |SHAP| by feature (test set, global importance):")
    for name, val in sorted(zip(FEATURE_NAMES, mean_abs_shap), key=lambda t: -t[1]):
        print(f"  {name:40s} {val:.4f}")

    # --- Save artifacts for inference (predict.py) ---
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(logreg, ARTIFACT_DIR / "logreg_model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(gb, ARTIFACT_DIR / "gb_model.joblib")
    joblib.dump({"lower": lower_bounds, "upper": upper_bounds}, ARTIFACT_DIR / "winsor_bounds.joblib")

    metadata = {
        "feature_names": FEATURE_NAMES,
        "n_training_rows": len(X_train),
        "n_test_rows": len(X_test),
        "positive_class_rate": float(y.mean()),
        "dataset_source": "UCI Taiwanese Bankruptcy Prediction (id=572), Taiwan Economic Journal 1999-2009",
        "logreg_test_auc": float(roc_auc_score(y_test, logreg_probs)),
        "gb_test_auc": float(roc_auc_score(y_test, gb_probs)),
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACT_DIR}/")
    print("Run models/ml_distress/predict.py <TICKER> to apply this model to a live company.")


def _report(model_name: str, y_test, preds, probs):
    print(f"\n--- {model_name} test-set performance (honest, given ~3% positive class) ---")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, probs):.4f}  (threshold-independent; the primary metric given imbalance)")
    print(f"  Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_test, preds, zero_division=0):.4f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]:\n{confusion_matrix(y_test, preds)}")


if __name__ == "__main__":
    train_and_evaluate()
