"""
models/credit_ratios.py

Phase 2 — Credit Risk Calculations
-----------------------------------
Pure, transparent functions that take standardized annual financial inputs
and return scores WITH the formula and the actual numbers plugged in, so
every output can be defended line-by-line in an interview.

Nothing here is a black box: every function returns a dict containing
{value/score, zone/interpretation, formula (string), inputs (dict)}.

Known, documented limitations (see module docstring sections below and the
app's Methodology tab):
  1. Classic Altman Z-Score needs MARKET value of equity. Free yfinance only
     exposes CURRENT market cap, not historical, so a proper 5-year Z-Score
     trend using market values would silently misstate older years. Classic
     Z-Score is therefore only computed for the latest fiscal year (using
     today's market cap), and is clearly labeled as such.
  2. Z''-Score uses BOOK value of equity instead, which IS available for
     every historical year — this is the score used for the 5-year trend
     chart in the dashboard.
  3. Ohlson O-Score originally divides Total Assets by a US GNP price-level
     deflator (base year 1968). No equivalent index is used here for
     non-US markets; instead, raw log(Total Assets) is used, which is the
     standard adaptation seen in academic replications outside the US.
     This is a deviation from Ohlson's original 1980 specification and is
     called out explicitly.
  4. Ohlson's "funds from operations" (FFO) term is proxied using Operating
     Cash Flow from the cash flow statement, a common and defensible
     simplification when FFO is not separately reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1: Standardized line-item extraction
#
# yfinance line-item labels are NOT perfectly consistent across companies/
# sectors (e.g. banks vs. manufacturers report different balance sheet
# structures). This map tries a list of candidate labels per standardized
# field and takes the first one found for each fiscal year.
# ---------------------------------------------------------------------------

LINE_ITEM_CANDIDATES = {
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "inventory": ["Inventory"],
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash Financial",
    ],
    "retained_earnings": ["Retained Earnings"],
    "total_debt": ["Total Debt"],
    "total_equity_book": [
        "Common Stock Equity",
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
    ],
    "ebit": ["EBIT"],
    "ebitda": ["EBITDA"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "total_revenue": ["Total Revenue", "Operating Revenue"],
    "interest_expense": ["Interest Expense"],
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
}


def _get(row: pd.Series, field: str) -> Optional[float]:
    """Look up a standardized field in a tidy-format row using the candidate label list."""
    for label in LINE_ITEM_CANDIDATES.get(field, []):
        if label in row.index:
            val = row[label]
            if pd.notna(val):
                return float(val)
    return None


def build_annual_inputs(
    income_statement_tidy: pd.DataFrame,
    balance_sheet_tidy: pd.DataFrame,
    cash_flow_tidy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the three tidy statements (years x line items) on fiscal year, and
    extract a standardized column per field defined in LINE_ITEM_CANDIDATES.

    Returns a DataFrame indexed by fiscal year (ascending) with one column
    per standardized field. Missing values are left as NaN — downstream
    ratio functions decide how to handle NaNs (usually: return None for
    that year with a note, never silently substitute zero).
    """
    all_years = sorted(
        set(income_statement_tidy.index)
        | set(balance_sheet_tidy.index)
        | set(cash_flow_tidy.index)
    )

    combined = pd.DataFrame(index=all_years, columns=list(LINE_ITEM_CANDIDATES.keys()), dtype=float)

    for year in all_years:
        merged_row = pd.Series(dtype=float)
        for df in (income_statement_tidy, balance_sheet_tidy, cash_flow_tidy):
            if year in df.index:
                merged_row = pd.concat([merged_row, df.loc[year]])

        for field in LINE_ITEM_CANDIDATES:
            combined.loc[year, field] = _get(merged_row, field)

    # Derived fields
    combined["working_capital"] = combined["current_assets"] - combined["current_liabilities"]

    return combined


# ---------------------------------------------------------------------------
# Step 2: Altman Z-Score (classic, public manufacturing firms)
# Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
# A = Working Capital / Total Assets
# B = Retained Earnings / Total Assets
# C = EBIT / Total Assets
# D = Market Value of Equity / Total Liabilities
# E = Sales / Total Assets
# Zones: Z > 2.99 Safe | 1.81-2.99 Grey | < 1.81 Distress
# ---------------------------------------------------------------------------

def altman_z_score(row: pd.Series, market_cap: float) -> dict:
    """
    Classic Altman Z-Score. Requires CURRENT market cap (see module docstring
    limitation #1) — only valid for the most recent fiscal year, since
    historical market cap isn't available from the free data source.
    """
    total_assets = row["total_assets"]
    total_liabilities = row["total_liabilities"]
    working_capital = row["working_capital"]
    retained_earnings = row["retained_earnings"]
    ebit = row["ebit"]
    sales = row["total_revenue"]

    required = [total_assets, total_liabilities, working_capital, retained_earnings, ebit, sales, market_cap]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in required):
        return {"score": None, "zone": "N/A", "note": "Insufficient data to compute classic Altman Z-Score for this year."}

    A = working_capital / total_assets
    B = retained_earnings / total_assets
    C = ebit / total_assets
    D = market_cap / total_liabilities
    E = sales / total_assets

    z = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E

    if z > 2.99:
        zone = "Safe"
    elif z >= 1.81:
        zone = "Grey"
    else:
        zone = "Distress"

    return {
        "score": round(z, 3),
        "zone": zone,
        "formula": "Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MVE/TL) + 1.0*(Sales/TA)",
        "inputs": {
            "Working Capital": working_capital, "Total Assets": total_assets,
            "Retained Earnings": retained_earnings, "EBIT": ebit,
            "Market Value of Equity (current)": market_cap, "Total Liabilities": total_liabilities,
            "Sales": sales,
        },
        "components": {"A (WC/TA)": round(A, 4), "B (RE/TA)": round(B, 4), "C (EBIT/TA)": round(C, 4),
                        "D (MVE/TL)": round(D, 4), "E (Sales/TA)": round(E, 4)},
        "note": "Uses TODAY's market cap — only valid for the most recent fiscal year, not a historical trend.",
    }


# ---------------------------------------------------------------------------
# Step 3: Altman Z''-Score (private firms / non-manufacturing, book-value based)
# Z'' = 6.56A + 3.26B + 6.72C + 1.05D
# D uses BOOK value of equity, not market value -> usable for every year.
# Zones: Z'' > 2.6 Safe | 1.1-2.6 Grey | < 1.1 Distress
# ---------------------------------------------------------------------------

def altman_z_double_prime_score(row: pd.Series) -> dict:
    total_assets = row["total_assets"]
    total_liabilities = row["total_liabilities"]
    working_capital = row["working_capital"]
    retained_earnings = row["retained_earnings"]
    ebit = row["ebit"]
    book_equity = row["total_equity_book"]

    required = [total_assets, total_liabilities, working_capital, retained_earnings, ebit, book_equity]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in required):
        return {"score": None, "zone": "N/A", "note": "Insufficient data to compute Z''-Score for this year."}

    A = working_capital / total_assets
    B = retained_earnings / total_assets
    C = ebit / total_assets
    D = book_equity / total_liabilities

    z_pp = 6.56 * A + 3.26 * B + 6.72 * C + 1.05 * D

    if z_pp > 2.6:
        zone = "Safe"
    elif z_pp >= 1.1:
        zone = "Grey"
    else:
        zone = "Distress"

    return {
        "score": round(z_pp, 3),
        "zone": zone,
        "formula": "Z'' = 6.56*(WC/TA) + 3.26*(RE/TA) + 6.72*(EBIT/TA) + 1.05*(BVE/TL)",
        "inputs": {
            "Working Capital": working_capital, "Total Assets": total_assets,
            "Retained Earnings": retained_earnings, "EBIT": ebit,
            "Book Value of Equity": book_equity, "Total Liabilities": total_liabilities,
        },
        "components": {"A (WC/TA)": round(A, 4), "B (RE/TA)": round(B, 4),
                        "C (EBIT/TA)": round(C, 4), "D (BVE/TL)": round(D, 4)},
        "note": "Uses BOOK equity, so this is valid for every historical year — used for the 5-year Z-Score trend chart.",
    }


# ---------------------------------------------------------------------------
# Step 4: Ohlson O-Score
# O = -1.32 -0.407*log(TA) +6.03*(TL/TA) -1.43*(WC/TA) +0.0757*(CL/CA)
#     -1.72*OENEG -2.37*(NI/TA) -1.83*(FFO/TL) +0.285*INTWO
#     -0.521*((NI_t - NI_t-1)/(|NI_t|+|NI_t-1|))
# P(distress) = 1 / (1 + e^-O)
# ---------------------------------------------------------------------------

def ohlson_o_score(row: pd.Series, prior_row: Optional[pd.Series]) -> dict:
    total_assets = row["total_assets"]
    total_liabilities = row["total_liabilities"]
    working_capital = row["working_capital"]
    current_liabilities = row["current_liabilities"]
    current_assets = row["current_assets"]
    net_income = row["net_income"]
    ffo_proxy = row["operating_cash_flow"]

    required = [total_assets, total_liabilities, working_capital, current_liabilities,
                current_assets, net_income, ffo_proxy]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) or (v <= 0 and False) for v in required):
        pass
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in required) or total_assets <= 0:
        return {"score": None, "probability": None, "note": "Insufficient data to compute Ohlson O-Score for this year."}

    prior_net_income = None
    if prior_row is not None:
        prior_net_income = prior_row.get("net_income")
        if prior_net_income is not None and isinstance(prior_net_income, float) and math.isnan(prior_net_income):
            prior_net_income = None

    oeneg = 1.0 if total_liabilities > total_assets else 0.0
    intwo = 1.0 if (net_income < 0 and prior_net_income is not None and prior_net_income < 0) else 0.0

    if prior_net_income is not None and (abs(net_income) + abs(prior_net_income)) != 0:
        ni_change_term = (net_income - prior_net_income) / (abs(net_income) + abs(prior_net_income))
    else:
        ni_change_term = 0.0  # no prior year available -> term contributes nothing, noted below

    log_ta = math.log(total_assets)

    o = (
        -1.32
        - 0.407 * log_ta
        + 6.03 * (total_liabilities / total_assets)
        - 1.43 * (working_capital / total_assets)
        + 0.0757 * (current_liabilities / current_assets if current_assets else 0)
        - 1.72 * oeneg
        - 2.37 * (net_income / total_assets)
        - 1.83 * (ffo_proxy / total_liabilities)
        + 0.285 * intwo
        - 0.521 * ni_change_term
    )

    probability = 1 / (1 + math.exp(-o))

    if probability >= 0.5:
        zone = "High distress probability"
    elif probability >= 0.2:
        zone = "Elevated risk"
    else:
        zone = "Low distress probability"

    return {
        "score": round(o, 4),
        "probability": round(probability, 4),
        "zone": zone,
        "formula": (
            "O = -1.32 -0.407*ln(TA) +6.03*(TL/TA) -1.43*(WC/TA) +0.0757*(CL/CA) "
            "-1.72*OENEG -2.37*(NI/TA) -1.83*(FFO/TL) +0.285*INTWO "
            "-0.521*((NI_t - NI_t-1)/(|NI_t|+|NI_t-1|));  P = 1/(1+e^-O)"
        ),
        "inputs": {
            "Total Assets": total_assets, "Total Liabilities": total_liabilities,
            "Working Capital": working_capital, "Current Liabilities": current_liabilities,
            "Current Assets": current_assets, "Net Income (current yr)": net_income,
            "Net Income (prior yr)": prior_net_income,
            "Operating Cash Flow (FFO proxy)": ffo_proxy,
            "OENEG (TL > TA)": oeneg, "INTWO (2 yrs neg. NI)": intwo,
        },
        "note": (
            "Uses raw ln(Total Assets) instead of Ohlson's original US GNP price-level "
            "deflator (no equivalent index applied here for non-US markets) — a standard, "
            "documented adaptation. FFO is proxied by Operating Cash Flow."
            + ("" if prior_net_income is not None else " No prior-year net income available; the year-over-year NI change term was set to 0.")
        ),
    }


# ---------------------------------------------------------------------------
# Step 5: Liquidity ratios
# ---------------------------------------------------------------------------

def liquidity_ratios(row: pd.Series) -> dict:
    ca, cl = row["current_assets"], row["current_liabilities"]
    inventory = row["inventory"]
    cash = row["cash_and_equivalents"]

    def safe_div(n, d):
        if n is None or d is None or (isinstance(d, float) and (math.isnan(d) or d == 0)) or (isinstance(n, float) and math.isnan(n)):
            return None
        return round(n / d, 3)

    current_ratio = safe_div(ca, cl)
    quick_ratio = safe_div((ca - inventory) if (ca is not None and inventory is not None) else None, cl)
    cash_ratio = safe_div(cash, cl)

    return {
        "current_ratio": {"value": current_ratio, "formula": "Current Assets / Current Liabilities",
                           "inputs": {"Current Assets": ca, "Current Liabilities": cl}},
        "quick_ratio": {"value": quick_ratio, "formula": "(Current Assets - Inventory) / Current Liabilities",
                         "inputs": {"Current Assets": ca, "Inventory": inventory, "Current Liabilities": cl}},
        "cash_ratio": {"value": cash_ratio, "formula": "Cash & Equivalents / Current Liabilities",
                        "inputs": {"Cash & Equivalents": cash, "Current Liabilities": cl}},
    }


# ---------------------------------------------------------------------------
# Step 6: Leverage ratios
# ---------------------------------------------------------------------------

def leverage_ratios(row: pd.Series) -> dict:
    total_debt = row["total_debt"]
    book_equity = row["total_equity_book"]
    ebitda = row["ebitda"]
    ebit = row["ebit"]
    interest_expense = row["interest_expense"]

    def safe_div(n, d):
        if n is None or d is None or (isinstance(d, float) and (math.isnan(d) or d == 0)) or (isinstance(n, float) and math.isnan(n)):
            return None
        return round(n / d, 3)

    debt_to_equity = safe_div(total_debt, book_equity)
    debt_to_ebitda = safe_div(total_debt, ebitda)
    # Interest expense is often reported as a negative number by yfinance; use abs() for the ratio.
    interest_coverage = safe_div(ebit, abs(interest_expense) if interest_expense is not None and not (isinstance(interest_expense, float) and math.isnan(interest_expense)) else None)

    return {
        "debt_to_equity": {"value": debt_to_equity, "formula": "Total Debt / Book Equity",
                            "inputs": {"Total Debt": total_debt, "Book Equity": book_equity}},
        "debt_to_ebitda": {"value": debt_to_ebitda, "formula": "Total Debt / EBITDA",
                            "inputs": {"Total Debt": total_debt, "EBITDA": ebitda}},
        "interest_coverage": {"value": interest_coverage, "formula": "EBIT / |Interest Expense|",
                               "inputs": {"EBIT": ebit, "Interest Expense": interest_expense}},
    }


# ---------------------------------------------------------------------------
# Step 7: Profitability trend (margin compression/expansion across years)
# ---------------------------------------------------------------------------

def profitability_trend(annual_inputs: pd.DataFrame) -> dict:
    """Net margin per year, plus direction of change across the available years."""
    margins = {}
    for year, row in annual_inputs.iterrows():
        revenue, ni = row["total_revenue"], row["net_income"]
        if revenue and not math.isnan(revenue) and revenue != 0 and ni is not None and not math.isnan(ni):
            margins[year] = round(ni / revenue, 4)

    if len(margins) < 2:
        return {"margins_by_year": margins, "trend": "Insufficient years to determine a trend."}

    years_sorted = sorted(margins.keys())
    first, last = margins[years_sorted[0]], margins[years_sorted[-1]]
    delta = round(last - first, 4)

    if delta > 0.005:
        trend = f"Margin expansion of {delta*100:.2f} pp from {years_sorted[0].year} to {years_sorted[-1].year}"
    elif delta < -0.005:
        trend = f"Margin compression of {abs(delta)*100:.2f} pp from {years_sorted[0].year} to {years_sorted[-1].year}"
    else:
        trend = "Net margin roughly flat over the period"

    return {
        "margins_by_year": margins,
        "formula": "Net Margin = Net Income / Total Revenue, per fiscal year",
        "trend": trend,
    }
