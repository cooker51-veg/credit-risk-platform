"""
data_pipeline/fetch_financials.py

Phase 1 — Data Pipeline
------------------------
Pulls up to 5 years of annual financial statements (Income Statement,
Balance Sheet, Cash Flow) for a given ticker using yfinance, cleans them,
and returns a structured, analysis-ready object.

Design notes:
- yfinance's free annual statements typically return 4 fiscal years of data
  (not always a full 5) depending on the exchange and how far back Yahoo
  Finance has data indexed. This is a known, documented limitation of the
  free data source — it is called out explicitly here and again in the
  app's Methodology tab so it never gets asserted as a bug during an
  interview.
- Indian tickers (NSE/BSE) need a suffix: '.NS' for NSE, '.BO' for BSE
  (e.g., 'RELIANCE.NS', 'TATASTEEL.BO').
- All three statements are returned as pandas DataFrames indexed by
  line-item, with columns as fiscal year-end dates (yfinance's native
  orientation), plus a "tidy" transposed version (years as rows) for
  ratio calculations downstream in Phase 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class FinancialDataError(Exception):
    """Raised when a ticker's financial data cannot be retrieved or is unusable."""


@dataclass
class CompanyFinancials:
    ticker: str
    company_name: str
    currency: str
    sector: Optional[str]
    industry: Optional[str]

    income_statement: pd.DataFrame   # raw yfinance orientation (line items x years)
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame

    income_statement_tidy: pd.DataFrame  # years x line items (rows = fiscal years, ascending)
    balance_sheet_tidy: pd.DataFrame
    cash_flow_tidy: pd.DataFrame

    years_available: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Company: {self.company_name} ({self.ticker})",
            f"Sector / Industry: {self.sector} / {self.industry}",
            f"Currency: {self.currency}",
            f"Fiscal years available: {', '.join(str(y) for y in self.years_available)}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_company_financials(ticker: str) -> CompanyFinancials:
    """
    Pull and clean 5 years (or as many as yfinance provides) of annual
    financial statements for `ticker`.

    Raises FinancialDataError with a clear message if the ticker is invalid
    or has no usable statement data.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise FinancialDataError("No ticker provided.")

    logger.info("Fetching data for %s", ticker)
    tk = yf.Ticker(ticker)

    # --- Basic identity / metadata ---
    try:
        info = tk.info or {}
    except Exception as e:
        raise FinancialDataError(
            f"Could not reach Yahoo Finance for '{ticker}'. This may be an invalid "
            f"ticker, a network issue, or rate limiting. Details: {e}"
        )

    company_name = info.get("longName") or info.get("shortName") or ticker
    currency = info.get("financialCurrency") or info.get("currency") or "N/A"
    sector = info.get("sector")
    industry = info.get("industry")

    if not info or (info.get("quoteType") is None and info.get("longName") is None):
        raise FinancialDataError(
            f"'{ticker}' does not appear to be a valid ticker. "
            f"For Indian equities, remember the exchange suffix "
            f"(e.g., 'RELIANCE.NS' for NSE, 'RELIANCE.BO' for BSE)."
        )

    # --- Pull the three annual statements ---
    warnings: list[str] = []

    income_statement = _safe_fetch(tk, "income_stmt", "annual income statement", warnings)
    balance_sheet = _safe_fetch(tk, "balance_sheet", "annual balance sheet", warnings)
    cash_flow = _safe_fetch(tk, "cash_flow", "annual cash flow statement", warnings)

    if income_statement.empty and balance_sheet.empty and cash_flow.empty:
        raise FinancialDataError(
            f"No financial statement data is available for '{ticker}' via yfinance. "
            f"This can happen for very small-cap, delisted, or non-equity tickers. "
            f"Try a large, actively-covered company to confirm the pipeline works."
        )

    # yfinance returns line items as rows, fiscal year-end dates as columns,
    # most-recent year first. Standardize column order to chronological
    # (oldest -> newest) for trend analysis, and cap at 5 years.
    income_statement = _standardize_columns(income_statement)
    balance_sheet = _standardize_columns(balance_sheet)
    cash_flow = _standardize_columns(cash_flow)

    n_years = max(
        income_statement.shape[1] if not income_statement.empty else 0,
        balance_sheet.shape[1] if not balance_sheet.empty else 0,
        cash_flow.shape[1] if not cash_flow.empty else 0,
    )
    if n_years < 5:
        warnings.append(
            f"Only {n_years} fiscal year(s) of data returned by yfinance's free "
            f"endpoint (requested 5). This is a known limitation of the free data "
            f"source, not a pipeline bug — Yahoo Finance typically indexes ~4 years "
            f"of annual statements for most tickers."
        )

    # Tidy (transposed) versions: rows = fiscal years ascending, columns = line items.
    # This orientation is far easier to feed into ratio/trend functions in Phase 2.
    income_statement_tidy = income_statement.transpose().sort_index()
    balance_sheet_tidy = balance_sheet.transpose().sort_index()
    cash_flow_tidy = cash_flow.transpose().sort_index()

    years_available = sorted(
        set(income_statement_tidy.index)
        | set(balance_sheet_tidy.index)
        | set(cash_flow_tidy.index)
    )

    if income_statement.empty:
        warnings.append("Income statement data unavailable for this ticker.")
    if balance_sheet.empty:
        warnings.append("Balance sheet data unavailable for this ticker.")
    if cash_flow.empty:
        warnings.append("Cash flow statement data unavailable for this ticker.")

    result = CompanyFinancials(
        ticker=ticker,
        company_name=company_name,
        currency=currency,
        sector=sector,
        industry=industry,
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        income_statement_tidy=income_statement_tidy,
        balance_sheet_tidy=balance_sheet_tidy,
        cash_flow_tidy=cash_flow_tidy,
        years_available=years_available,
        warnings=warnings,
    )

    logger.info("Fetched %s: %d fiscal years, %d warning(s)", ticker, n_years, len(warnings))
    return result


def _safe_fetch(tk: yf.Ticker, attr: str, label: str, warnings: list) -> pd.DataFrame:
    """Fetch a statement attribute off a yfinance Ticker object, never raising."""
    try:
        df = getattr(tk, attr)
        if df is None or df.empty:
            warnings.append(f"No {label} data returned.")
            return pd.DataFrame()
        return df
    except Exception as e:
        warnings.append(f"Error fetching {label}: {e}")
        return pd.DataFrame()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort columns chronologically (oldest -> newest), cap at most recent 5 years."""
    if df.empty:
        return df
    df = df.copy()
    df = df.loc[:, sorted(df.columns)]  # chronological order
    if df.shape[1] > 5:
        df = df.iloc[:, -5:]  # keep most recent 5
    return df


# ---------------------------------------------------------------------------
# Manual verification entry point (Phase 1 requirement:
# "print/display the raw output so I can verify it's correct")
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"

    print("=" * 80)
    print(f"FETCHING: {test_ticker}")
    print("=" * 80)

    try:
        data = fetch_company_financials(test_ticker)
    except FinancialDataError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("\n" + data.summary())

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 80)

    print("\n" + "-" * 80)
    print("INCOME STATEMENT (raw, line items x fiscal years)")
    print("-" * 80)
    print(data.income_statement)

    print("\n" + "-" * 80)
    print("BALANCE SHEET (raw, line items x fiscal years)")
    print("-" * 80)
    print(data.balance_sheet)

    print("\n" + "-" * 80)
    print("CASH FLOW STATEMENT (raw, line items x fiscal years)")
    print("-" * 80)
    print(data.cash_flow)

    print("\n" + "-" * 80)
    print("TIDY INCOME STATEMENT (fiscal years x line items) — used downstream")
    print("-" * 80)
    print(data.income_statement_tidy)
