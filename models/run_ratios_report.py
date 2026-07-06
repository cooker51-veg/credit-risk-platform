"""
models/run_ratios_report.py

Run this LOCALLY (not in the sandbox) to verify Phase 2 against real data:

    python models/run_ratios_report.py RELIANCE.NS

Prints, for every available fiscal year: Altman Z''-Score, classic Altman
Z-Score (latest year only), Ohlson O-Score, liquidity ratios, leverage
ratios, and the profitability trend — each with the formula and the actual
numbers plugged in, so you can check every figure by hand.
"""

import sys

from data_pipeline.fetch_financials import fetch_company_financials, FinancialDataError
from models.credit_ratios import (
    build_annual_inputs,
    altman_z_score,
    altman_z_double_prime_score,
    ohlson_o_score,
    liquidity_ratios,
    leverage_ratios,
    profitability_trend,
)


def _print_block(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main(ticker: str):
    try:
        financials = fetch_company_financials(ticker)
    except FinancialDataError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(financials.summary())

    annual_inputs = build_annual_inputs(
        financials.income_statement_tidy,
        financials.balance_sheet_tidy,
        financials.cash_flow_tidy,
    )

    years = list(annual_inputs.index)
    if not years:
        print("No usable fiscal years found — cannot compute ratios.")
        return

    # --- Z''-Score trend across ALL years (book-value based) ---
    _print_block("ALTMAN Z''-SCORE (5-YEAR TREND, book-value based)")
    for year in years:
        result = altman_z_double_prime_score(annual_inputs.loc[year])
        print(f"\nFY {year.date()}:")
        if result["score"] is None:
            print(f"  {result['note']}")
            continue
        print(f"  Formula: {result['formula']}")
        print(f"  Inputs: {result['inputs']}")
        print(f"  Components: {result['components']}")
        print(f"  Z'' = {result['score']}  ->  Zone: {result['zone']}")

    # --- Classic Z-Score, latest year only (needs current market cap) ---
    _print_block("ALTMAN Z-SCORE (CLASSIC — most recent fiscal year only)")
    latest_year = years[-1]
    try:
        import yfinance as yf
        market_cap = yf.Ticker(financials.ticker).info.get("marketCap")
    except Exception:
        market_cap = None

    if market_cap:
        result = altman_z_score(annual_inputs.loc[latest_year], market_cap)
        print(f"FY {latest_year.date()} (using TODAY's market cap: {market_cap:,.0f}):")
        if result["score"] is not None:
            print(f"  Formula: {result['formula']}")
            print(f"  Inputs: {result['inputs']}")
            print(f"  Components: {result['components']}")
            print(f"  Z = {result['score']}  ->  Zone: {result['zone']}")
            print(f"  NOTE: {result['note']}")
        else:
            print(f"  {result['note']}")
    else:
        print("Current market cap unavailable — skipping classic Z-Score.")

    # --- Ohlson O-Score across years with a prior year available ---
    _print_block("OHLSON O-SCORE")
    for i, year in enumerate(years):
        if i == 0:
            continue  # no prior year to compare against
        prior_row = annual_inputs.loc[years[i - 1]]
        result = ohlson_o_score(annual_inputs.loc[year], prior_row)
        print(f"\nFY {year.date()}:")
        if result["score"] is None:
            print(f"  {result['note']}")
            continue
        print(f"  Formula: {result['formula']}")
        print(f"  Inputs: {result['inputs']}")
        print(f"  O = {result['score']}  ->  P(distress) = {result['probability']}  ->  {result['zone']}")
        print(f"  NOTE: {result['note']}")

    # --- Liquidity & leverage, latest year ---
    _print_block(f"LIQUIDITY RATIOS (FY {latest_year.date()})")
    liq = liquidity_ratios(annual_inputs.loc[latest_year])
    for name, d in liq.items():
        print(f"  {name}: {d['value']}  |  {d['formula']}  |  {d['inputs']}")

    _print_block(f"LEVERAGE RATIOS (FY {latest_year.date()})")
    lev = leverage_ratios(annual_inputs.loc[latest_year])
    for name, d in lev.items():
        print(f"  {name}: {d['value']}  |  {d['formula']}  |  {d['inputs']}")

    # --- Profitability trend ---
    _print_block("PROFITABILITY TREND")
    trend = profitability_trend(annual_inputs)
    print(f"  Formula: {trend.get('formula')}")
    print(f"  Margins by year: {trend['margins_by_year']}")
    print(f"  {trend['trend']}")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    main(t)
