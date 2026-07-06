# Verifying Phase 1 locally

This build sandbox can't reach `query1/query2.finance.yahoo.com`, so the live
data pull needs to be verified on your own machine. This takes under 2 minutes.

```bash
cd credit_risk_platform
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Print raw + tidy statements for a ticker (default is RELIANCE.NS)
python3 data_pipeline/fetch_financials.py RELIANCE.NS

# Try a couple of others to stress-test edge cases:
python3 data_pipeline/fetch_financials.py TCS.NS
python3 data_pipeline/fetch_financials.py HDFCBANK.NS
python3 data_pipeline/fetch_financials.py TATASTEEL.BO
```

## What to check in the output

1. **`summary()` block** — correct company name, sector, currency, and how
   many fiscal years actually came back (usually 4, sometimes 5 — this is a
   yfinance/Yahoo limitation, flagged automatically in `warnings` when it
   happens).
2. **Raw income statement / balance sheet / cash flow** — line items should
   look like standard IFRS/Ind-AS-style labels (`Total Revenue`, `EBITDA`,
   `Net Income`, `Total Assets`, `Total Debt`, etc.). Some smaller companies
   will have sparser line items — that's Yahoo's data, not a bug.
3. **Tidy income statement** — same data, just transposed so fiscal years are
   rows in ascending order. This is what Phase 2's ratio functions will
   consume directly.

## Run the transform-logic test (no network needed)

```bash
PYTHONPATH=. python3 tests/test_fetch_financials_mock.py
```

This validates column ordering, the 5-year cap, and the tidy transpose using
fabricated data — it already passes in the build sandbox and should pass
identically for you.

## If a ticker fails

- Double check the exchange suffix: `.NS` for NSE, `.BO` for BSE.
- Yahoo occasionally rate-limits repeated rapid requests — wait ~30s and retry.
- Very small/illiquid tickers sometimes have incomplete statement coverage;
  that's expected and is exactly what the `warnings` list is for.

Once you've confirmed real output looks right, tell me and we'll move to
**Phase 2: Credit Risk Calculations** (Altman Z-Score, Z''-Score, Ohlson
O-Score, liquidity/leverage/profitability ratios).
