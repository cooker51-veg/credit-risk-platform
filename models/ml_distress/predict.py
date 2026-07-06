"""
models/ml_distress/predict.py

Applies the trained distress model (from train.py's saved artifacts) to a
live company pulled via yfinance, and explains the prediction with SHAP.

Usage (run locally, after train.py has produced artifacts/):
    python models/ml_distress/predict.py RELIANCE.NS
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from data_pipeline.fetch_financials import fetch_company_financials, FinancialDataError
from models.credit_ratios import build_annual_inputs
from .features import FEATURE_NAMES, features_from_annual_inputs, rescale_live_features_to_uci_band

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "logreg_model.joblib")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    winsor_bounds = joblib.load(ARTIFACT_DIR / "winsor_bounds.joblib")
    with open(ARTIFACT_DIR / "metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, winsor_bounds, metadata


def predict_distress(ticker: str):
    model, scaler, winsor_bounds, metadata = load_artifacts()

    try:
        financials = fetch_company_financials(ticker)
    except FinancialDataError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    annual_inputs = build_annual_inputs(
        financials.income_statement_tidy,
        financials.balance_sheet_tidy,
        financials.cash_flow_tidy,
    )

    print(f"\n{financials.company_name} ({ticker})")
    print("=" * 70)
    print(
        f"Model trained on: {metadata['dataset_source']}\n"
        f"Training positive class rate: {metadata['positive_class_rate']*100:.2f}%  |  "
        f"Test ROC-AUC: {metadata['logreg_test_auc']:.3f}\n"
        f"REMINDER: this dataset is Taiwanese companies, 1999-2009 — output below is a "
        f"directional distress signal from pattern-matching on 10 ratios, NOT a probability "
        f"calibrated to real-world Indian default rates.\n"
    )

    any_computed = False
    for year, row in annual_inputs.iterrows():
        feature_vals = features_from_annual_inputs(row)
        if feature_vals is None:
            print(f"FY {year.date()}: insufficient data to compute all 10 features — skipped.")
            continue

        any_computed = True
        rescaled_vals = rescale_live_features_to_uci_band(feature_vals)
        X = np.array(rescaled_vals).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]

        explainer = shap.LinearExplainer(model, scaler.transform(np.zeros((1, len(FEATURE_NAMES)))))
        # Use training background implicitly via model's fitted coefficients; for a single
        # instance explanation we compute contribution = coef * standardized_value.
        contributions = model.coef_[0] * X_scaled[0]

        print(f"\nFY {year.date()}: Distress probability = {prob*100:.1f}%")
        print(f"  Feature values and SHAP-style contributions (coef * standardized value):")
        for name, raw_val, contrib in sorted(
            zip(FEATURE_NAMES, feature_vals, contributions), key=lambda t: -abs(t[2])
        ):
            direction = "-> pushes toward distress" if contrib > 0 else "-> pushes toward safety"
            print(f"    {name:38s} raw={raw_val:+.4f}   contribution={contrib:+.4f}  {direction}")

    if not any_computed:
        print("\nCould not compute the ML distress score for any fiscal year — "
              "insufficient underlying data from yfinance for this ticker.")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    predict_distress(t)
