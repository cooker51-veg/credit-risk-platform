"""
tests/test_credit_ratios_mock.py

Unit tests for Phase 2 (models/credit_ratios.py) using hand-picked, round
numbers so the expected results can be verified by hand or calculator.
Expected values are computed via an INDEPENDENT formula restatement inside
each test (not just re-calling the module's own function), so these tests
actually catch implementation bugs rather than just confirming the code
agrees with itself.
"""

import math
import pandas as pd

from models.credit_ratios import (
    altman_z_score,
    altman_z_double_prime_score,
    ohlson_o_score,
    liquidity_ratios,
    leverage_ratios,
    profitability_trend,
)


def _mock_row(**overrides) -> pd.Series:
    base = {
        "total_assets": 1000.0,
        "total_liabilities": 500.0,
        "current_assets": 400.0,
        "current_liabilities": 200.0,
        "inventory": 100.0,
        "cash_and_equivalents": 50.0,
        "retained_earnings": 300.0,
        "total_debt": 400.0,
        "total_equity_book": 500.0,
        "ebit": 150.0,
        "ebitda": 200.0,
        "net_income": 50.0,
        "total_revenue": 1200.0,
        "interest_expense": -30.0,   # yfinance often reports this as negative
        "operating_cash_flow": 100.0,
    }
    base["working_capital"] = base["current_assets"] - base["current_liabilities"]
    base.update(overrides)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# Altman Z-Score (classic)
# ---------------------------------------------------------------------------

def test_altman_z_score_matches_hand_calculation():
    row = _mock_row()
    market_cap = 800.0

    # Independent hand calculation:
    # A = WC/TA = 200/1000 = 0.2
    # B = RE/TA = 300/1000 = 0.3
    # C = EBIT/TA = 150/1000 = 0.15
    # D = MVE/TL = 800/500 = 1.6
    # E = Sales/TA = 1200/1000 = 1.2
    # Z = 1.2*0.2 + 1.4*0.3 + 3.3*0.15 + 0.6*1.6 + 1.0*1.2
    #   = 0.24 + 0.42 + 0.495 + 0.96 + 1.2 = 3.315
    expected_z = 1.2*0.2 + 1.4*0.3 + 3.3*0.15 + 0.6*1.6 + 1.0*1.2

    result = altman_z_score(row, market_cap)

    assert result["score"] == round(expected_z, 3), f"Expected {expected_z}, got {result['score']}"
    assert abs(result["score"] - 3.315) < 1e-9
    assert result["zone"] == "Safe"  # > 2.99


def test_altman_z_score_distress_zone():
    # Push into distress territory: heavy liabilities, thin equity, weak EBIT
    row = _mock_row(retained_earnings=-200.0, ebit=10.0, total_liabilities=1200.0)
    result = altman_z_score(row, market_cap=100.0)
    assert result["score"] is not None
    assert result["zone"] == "Distress"


def test_altman_z_score_missing_market_cap_data_handled():
    row = _mock_row(ebit=None)
    result = altman_z_score(row, market_cap=800.0)
    assert result["score"] is None
    assert result["zone"] == "N/A"


# ---------------------------------------------------------------------------
# Altman Z''-Score (book-value based)
# ---------------------------------------------------------------------------

def test_altman_z_double_prime_matches_hand_calculation():
    row = _mock_row()

    # A = 0.2, B = 0.3, C = 0.15 (same as above)
    # D'' = Book Equity / TL = 500/500 = 1.0
    # Z'' = 6.56*0.2 + 3.26*0.3 + 6.72*0.15 + 1.05*1.0
    #     = 1.312 + 0.978 + 1.008 + 1.05 = 4.348
    expected_zpp = 6.56*0.2 + 3.26*0.3 + 6.72*0.15 + 1.05*1.0

    result = altman_z_double_prime_score(row)

    assert abs(result["score"] - round(expected_zpp, 3)) < 1e-9
    assert result["zone"] == "Safe"  # > 2.6


# ---------------------------------------------------------------------------
# Ohlson O-Score
# ---------------------------------------------------------------------------

def test_ohlson_o_score_matches_hand_calculation():
    row = _mock_row(net_income=50.0)
    prior_row = _mock_row(net_income=40.0)

    TA, TL, WC, CL, CA = 1000.0, 500.0, 200.0, 200.0, 400.0
    NI, NI_prior, FFO = 50.0, 40.0, 100.0

    oeneg = 0.0  # TL (500) < TA (1000)
    intwo = 0.0  # NI is positive, not two consecutive negative years
    ni_change_term = (NI - NI_prior) / (abs(NI) + abs(NI_prior))  # (10)/(90)

    expected_o = (
        -1.32
        - 0.407 * math.log(TA)
        + 6.03 * (TL / TA)
        - 1.43 * (WC / TA)
        + 0.0757 * (CL / CA)
        - 1.72 * oeneg
        - 2.37 * (NI / TA)
        - 1.83 * (FFO / TL)
        + 0.285 * intwo
        - 0.521 * ni_change_term
    )
    expected_prob = 1 / (1 + math.exp(-expected_o))

    result = ohlson_o_score(row, prior_row)

    assert abs(result["score"] - round(expected_o, 4)) < 1e-6
    assert abs(result["probability"] - round(expected_prob, 4)) < 1e-6


def test_ohlson_o_score_no_prior_year_sets_ni_change_term_to_zero():
    row = _mock_row()
    result = ohlson_o_score(row, prior_row=None)
    assert result["score"] is not None
    assert "No prior-year net income" in result["note"]


def test_ohlson_oeneg_flag_when_liabilities_exceed_assets():
    row = _mock_row(total_liabilities=1500.0)  # TL > TA (1000)
    result = ohlson_o_score(row, prior_row=_mock_row(net_income=40.0))
    assert result["inputs"]["OENEG (TL > TA)"] == 1.0


# ---------------------------------------------------------------------------
# Liquidity ratios
# ---------------------------------------------------------------------------

def test_liquidity_ratios_matches_hand_calculation():
    row = _mock_row()  # CA=400, CL=200, Inventory=100, Cash=50
    result = liquidity_ratios(row)

    assert result["current_ratio"]["value"] == round(400/200, 3)   # 2.0
    assert result["quick_ratio"]["value"] == round((400-100)/200, 3)  # 1.5
    assert result["cash_ratio"]["value"] == round(50/200, 3)       # 0.25


def test_liquidity_ratios_handles_zero_current_liabilities():
    row = _mock_row(current_liabilities=0.0)
    result = liquidity_ratios(row)
    assert result["current_ratio"]["value"] is None  # must not divide by zero


# ---------------------------------------------------------------------------
# Leverage ratios
# ---------------------------------------------------------------------------

def test_leverage_ratios_matches_hand_calculation():
    row = _mock_row()  # Debt=400, Equity=500, EBITDA=200, EBIT=150, IntExp=-30
    result = leverage_ratios(row)

    assert result["debt_to_equity"]["value"] == round(400/500, 3)   # 0.8
    assert result["debt_to_ebitda"]["value"] == round(400/200, 3)   # 2.0
    assert result["interest_coverage"]["value"] == round(150/30, 3)  # 5.0 (uses abs of -30)


# ---------------------------------------------------------------------------
# Profitability trend
# ---------------------------------------------------------------------------

def test_profitability_trend_detects_margin_expansion():
    df = pd.DataFrame({
        "total_revenue": [1000.0, 1100.0, 1200.0],
        "net_income": [50.0, 77.0, 108.0],   # margins: 5.0%, 7.0%, 9.0%
    }, index=pd.to_datetime(["2023-03-31", "2024-03-31", "2025-03-31"]))

    result = profitability_trend(df)
    assert "expansion" in result["trend"].lower()
    assert result["margins_by_year"][pd.Timestamp("2023-03-31")] == 0.05
    assert result["margins_by_year"][pd.Timestamp("2025-03-31")] == 0.09


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} tests passed.")
