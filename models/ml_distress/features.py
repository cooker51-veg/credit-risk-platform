"""
models/ml_distress/features.py

Defines the CANONICAL feature set used by the ML distress model — the same
10 ratios are computed identically whether the data comes from:
  (a) the UCI Taiwanese Bankruptcy Prediction dataset (training), or
  (b) a live company pulled via yfinance + Phase 2's build_annual_inputs
      (inference).

This consistency is the whole point: a model trained on mismatched features
from training vs. inference would be silently meaningless. Keeping one
canonical definition in one place prevents that.

Each feature is chosen because it exists (under some label) in BOTH the UCI
dataset's 95 raw columns AND is directly computable from the standardized
fields already extracted in models/credit_ratios.py's build_annual_inputs().
"""

from __future__ import annotations
import math
from typing import Optional
import pandas as pd

# Canonical feature names, in fixed order. This exact order/name list is the
# model's contract — training and inference must both produce this shape.
FEATURE_NAMES = [
    "working_capital_to_total_assets",
    "retained_earnings_to_total_assets",
    "roa_net_income_to_total_assets",
    "current_ratio",
    "quick_ratio",
    "debt_ratio_tl_to_ta",
    "net_worth_to_total_assets",
    "interest_coverage_ebit_to_interest",
    "debt_to_equity",
    "operating_margin",
]

# IMPORTANT — SCALE MISMATCH FIX:
# The UCI Taiwanese dataset's published columns are pre-squeezed into a
# roughly [0,1] band by the original researchers using unknown internal
# min/max values (confirmed by inspecting summary stats: e.g.
# working_capital_to_total_assets has mean=0.81, std=0.06, bounded [0,1] --
# not a raw ratio's natural range). A live company's genuine raw ratio
# (e.g. current ratio = 1.1) looks like an extreme outlier against that
# artificially tight band, causing the model to saturate near 100%
# regardless of actual financial health.
#
# Since the original normalization parameters were never published and
# cannot be inverted, live ratios are instead rescaled into a comparable
# [0,1] band using standard finance-textbook typical ranges before being
# fed into the (unchanged) trained model. This is an approximation, not an
# exact inverse transform, and is disclosed as such in the app's
# Methodology tab. It preserves each feature's DIRECTION (higher machine
# distress risk features still map to a higher post-transform value) while
# preventing raw live-company scale from blowing up the model's input.
DOMAIN_RESCALE_BOUNDS = {
    "working_capital_to_total_assets": (-0.3, 0.6),
    "retained_earnings_to_total_assets": (-1.0, 1.0),
    "roa_net_income_to_total_assets": (-0.3, 0.3),
    "current_ratio": (0.0, 4.0),
    "quick_ratio": (0.0, 3.0),
    "debt_ratio_tl_to_ta": (0.0, 1.5),
    "net_worth_to_total_assets": (-0.5, 1.0),
    "interest_coverage_ebit_to_interest": (-5.0, 20.0),
    "debt_to_equity": (0.0, 5.0),
    "operating_margin": (-0.5, 0.5),
}


def rescale_live_features_to_uci_band(raw_values: list) -> list:
    """
    Rescales a raw live-company feature vector (in FEATURE_NAMES order) into
    a [0,1] band comparable to the UCI training data's squeezed scale, using
    the domain-typical bounds above. Values outside the assumed typical
    range are clipped to 0 or 1.
    """
    rescaled = []
    for name, val in zip(FEATURE_NAMES, raw_values):
        lo, hi = DOMAIN_RESCALE_BOUNDS[name]
        scaled = (val - lo) / (hi - lo)
        rescaled.append(max(0.0, min(1.0, scaled)))
    return rescaled

# Raw UCI Taiwanese Bankruptcy Prediction column names (as published) that
# correspond to each canonical feature. Some UCI columns are already exactly
# these ratios; noted where a direct rename vs. a light transform is used.
UCI_COLUMN_MAP = {
    "working_capital_to_total_assets": " Working Capital to Total Assets",
    "retained_earnings_to_total_assets": " Retained Earnings to Total Assets",
    "roa_net_income_to_total_assets": " ROA(C) before interest and depreciation before interest",
    "current_ratio": " Current Ratio",
    "quick_ratio": " Quick Ratio",
    "debt_ratio_tl_to_ta": " Debt ratio %",
    "net_worth_to_total_assets": " Net worth/Assets",
    "interest_coverage_ebit_to_interest": " Interest Coverage Ratio (Interest expense to EBIT)",
    "debt_to_equity": " Total debt/Total net worth",
    "operating_margin": " Operating Profit Rate",
}


def features_from_uci_row(row: pd.Series) -> list:
    """Extract the canonical feature vector from a raw UCI dataset row."""
    return [float(row[UCI_COLUMN_MAP[f]]) for f in FEATURE_NAMES]


def features_from_annual_inputs(row: pd.Series) -> Optional[list]:
    """
    Extract the canonical feature vector from a Phase 2 `annual_inputs` row
    (built by models.credit_ratios.build_annual_inputs). Returns None if any
    required primitive is missing for this fiscal year.
    """
    ta = row.get("total_assets")
    tl = row.get("total_liabilities")
    wc = row.get("working_capital")
    re = row.get("retained_earnings")
    ni = row.get("net_income")
    ca = row.get("current_assets")
    cl = row.get("current_liabilities")
    inv = row.get("inventory")
    equity = row.get("total_equity_book")
    ebit = row.get("ebit")
    interest_exp = row.get("interest_expense")
    debt = row.get("total_debt")
    revenue = row.get("total_revenue")

    required = [ta, tl, wc, re, ni, ca, cl, equity, ebit, debt, revenue]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in required) or ta in (0, None) or ca in (0, None):
        return None

    wc_ta = wc / ta
    re_ta = re / ta
    roa = ni / ta
    current_ratio = ca / cl if cl else None
    quick_ratio = (ca - inv) / cl if (cl and inv is not None and not (isinstance(inv, float) and math.isnan(inv))) else None
    debt_ratio = tl / ta
    net_worth_ta = equity / ta
    interest_coverage = ebit / abs(interest_exp) if (interest_exp not in (None, 0) and not (isinstance(interest_exp, float) and math.isnan(interest_exp))) else None
    debt_equity = debt / equity if equity else None
    op_margin = ebit / revenue if revenue else None

    values = [wc_ta, re_ta, roa, current_ratio, quick_ratio, debt_ratio,
              net_worth_ta, interest_coverage, debt_equity, op_margin]

    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
        return None

    return values
