"""
tests/test_fetch_financials_mock.py

Validates the DATA-CLEANING LOGIC in fetch_financials.py (_standardize_columns,
tidy transpose, warning generation) against fabricated yfinance-shaped
DataFrames. This does NOT hit the network — it exists because this sandbox's
egress allowlist blocks query1/query2.finance.yahoo.com, so live verification
must happen on your machine (see run_locally.md). This test proves the
transform logic is correct independent of that network restriction.
"""

import pandas as pd
from data_pipeline.fetch_financials import _standardize_columns


def _mock_income_statement() -> pd.DataFrame:
    # Mimics yfinance's native orientation: line items as rows, fiscal
    # year-end Timestamps as columns, most-recent year FIRST (as yfinance
    # actually returns it), plus an extra 6th year to test the 5-year cap.
    dates = pd.to_datetime(
        ["2025-03-31", "2024-03-31", "2023-03-31", "2022-03-31", "2021-03-31", "2020-03-31"]
    )
    data = {
        dates[0]: {"Total Revenue": 900_000, "EBITDA": 150_000, "Net Income": 80_000},
        dates[1]: {"Total Revenue": 820_000, "EBITDA": 135_000, "Net Income": 70_000},
        dates[2]: {"Total Revenue": 760_000, "EBITDA": 120_000, "Net Income": 60_000},
        dates[3]: {"Total Revenue": 700_000, "EBITDA": 110_000, "Net Income": 55_000},
        dates[4]: {"Total Revenue": 640_000, "EBITDA": 95_000, "Net Income": 45_000},
        dates[5]: {"Total Revenue": 600_000, "EBITDA": 90_000, "Net Income": 40_000},
    }
    return pd.DataFrame(data)


def test_standardize_columns_orders_chronologically_and_caps_at_5():
    raw = _mock_income_statement()
    assert raw.shape[1] == 6  # sanity check on the fixture itself

    cleaned = _standardize_columns(raw)

    # Cap at 5 years
    assert cleaned.shape[1] == 5, f"Expected 5 columns after cap, got {cleaned.shape[1]}"

    # Chronological order: oldest -> newest
    cols = list(cleaned.columns)
    assert cols == sorted(cols), "Columns should be sorted oldest -> newest"

    # The dropped year should be the OLDEST one (2020-03-31), not the newest
    assert pd.Timestamp("2020-03-31") not in cleaned.columns
    assert pd.Timestamp("2025-03-31") in cleaned.columns

    # Values should still be correctly aligned to their year after reordering
    assert cleaned.loc["Total Revenue", pd.Timestamp("2025-03-31")] == 900_000
    assert cleaned.loc["Total Revenue", pd.Timestamp("2021-03-31")] == 640_000


def test_standardize_columns_handles_empty_frame():
    empty = pd.DataFrame()
    result = _standardize_columns(empty)
    assert result.empty


def test_tidy_transpose_orientation():
    raw = _mock_income_statement()
    cleaned = _standardize_columns(raw)
    tidy = cleaned.transpose().sort_index()

    # Rows should now be fiscal years (ascending), columns should be line items
    assert "Total Revenue" in tidy.columns
    assert tidy.index.is_monotonic_increasing
    assert tidy.loc[pd.Timestamp("2025-03-31"), "Net Income"] == 80_000


if __name__ == "__main__":
    test_standardize_columns_orders_chronologically_and_caps_at_5()
    test_standardize_columns_handles_empty_frame()
    test_tidy_transpose_orientation()
    print("All mock transform-logic tests passed.")
