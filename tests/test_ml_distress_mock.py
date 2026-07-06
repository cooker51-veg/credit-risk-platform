"""
tests/test_ml_distress_mock.py

Validates the ML DISTRESS PIPELINE MECHANICS using a synthetic dataset that
mimics the UCI Taiwanese Bankruptcy dataset's structure (10 canonical
features, ~3% positive class, same library calls as train.py). This does
NOT test whether the model is a GOOD predictor of real bankruptcy -- that
can only be judged against the real dataset, downloaded on your machine.
This proves the pipeline itself (scaling, imbalance handling, SHAP shapes,
save/reload) works correctly.

Also tests features.py's extraction logic against mock annual_inputs rows.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.ml_distress.features import FEATURE_NAMES, features_from_annual_inputs, rescale_live_features_to_uci_band


def _synthetic_dataset(n=2000, seed=42):
    """Mimics UCI dataset shape: 10 features, ~3% positive class, with a
    real (if simplistic) signal so ROC-AUC should be meaningfully > 0.5."""
    rng = np.random.default_rng(seed)
    n_features = len(FEATURE_NAMES)

    # "Healthy" companies: ratios centered favorably
    n_bankrupt = int(n * 0.03)
    n_healthy = n - n_bankrupt

    healthy = rng.normal(loc=0.5, scale=0.2, size=(n_healthy, n_features))
    bankrupt = rng.normal(loc=-0.3, scale=0.3, size=(n_bankrupt, n_features))

    X = np.vstack([healthy, bankrupt])
    y = np.array([0] * n_healthy + [1] * n_bankrupt)

    # shuffle
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    return pd.DataFrame(X, columns=FEATURE_NAMES), pd.Series(y, name="bankrupt")


def test_synthetic_pipeline_end_to_end():
    X, y = _synthetic_dataset()
    assert abs(y.mean() - 0.03) < 0.01, "Fixture should mimic ~3% positive class rate"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Stratification should roughly preserve class balance in both splits
    assert abs(y_train.mean() - y_test.mean()) < 0.02

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    auc = roc_auc_score(y_test, probs)
    assert auc > 0.6, f"Expected the synthetic signal to be learnable (AUC>0.6), got {auc:.3f}"

    assert model.coef_[0].shape == (len(FEATURE_NAMES),)


def test_shap_output_shape_matches_features():
    X, y = _synthetic_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    explainer = shap.LinearExplainer(model, X_scaled[:200])
    shap_values = explainer.shap_values(X_scaled[:50])

    assert shap_values.shape == (50, len(FEATURE_NAMES))


def test_model_save_and_reload_gives_identical_predictions(tmp_path):
    import joblib

    X, y = _synthetic_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    original_preds = model.predict_proba(X_scaled[:10])

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    reloaded = joblib.load(model_path)

    reloaded_preds = reloaded.predict_proba(X_scaled[:10])
    assert np.allclose(original_preds, reloaded_preds)


# ---------------------------------------------------------------------------
# features.py extraction logic
# ---------------------------------------------------------------------------

def _mock_annual_row(**overrides):
    base = {
        "total_assets": 1000.0, "total_liabilities": 500.0,
        "working_capital": 200.0, "retained_earnings": 300.0,
        "net_income": 50.0, "current_assets": 400.0, "current_liabilities": 200.0,
        "inventory": 100.0, "total_equity_book": 500.0, "ebit": 150.0,
        "interest_expense": -30.0, "total_debt": 400.0, "total_revenue": 1200.0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_features_from_annual_inputs_correct_values():
    row = _mock_annual_row()
    feats = features_from_annual_inputs(row)
    assert feats is not None
    assert len(feats) == len(FEATURE_NAMES)

    expected = {
        "working_capital_to_total_assets": 200/1000,
        "retained_earnings_to_total_assets": 300/1000,
        "roa_net_income_to_total_assets": 50/1000,
        "current_ratio": 400/200,
        "quick_ratio": (400-100)/200,
        "debt_ratio_tl_to_ta": 500/1000,
        "net_worth_to_total_assets": 500/1000,
        "interest_coverage_ebit_to_interest": 150/30,
        "debt_to_equity": 400/500,
        "operating_margin": 150/1200,
    }
    for name, val, exp in zip(FEATURE_NAMES, feats, expected.values()):
        assert abs(val - exp) < 1e-9, f"{name}: expected {exp}, got {val}"


def test_features_from_annual_inputs_returns_none_when_data_missing():
    row = _mock_annual_row(ebit=None)
    feats = features_from_annual_inputs(row)
    assert feats is None


def test_features_from_annual_inputs_handles_zero_current_liabilities():
    row = _mock_annual_row(current_liabilities=0.0)
    feats = features_from_annual_inputs(row)
    assert feats is None  # current_ratio would be undefined -> must not crash, must return None


def test_winsorization_neutralizes_outlier_distortion():
    """
    Reproduces the real bug found when training on the UCI dataset: one
    extreme outlier row in a ratio column (e.g. Current Ratio in the
    thousands, likely a data error) blows up StandardScaler's variance
    estimate and downstream SHAP/coefficient-contribution magnitude for
    every row, even though the fitted coefficient itself stays small.
    Clipping at the 1st/99th percentile (fit on train only) should bring
    the scaled outlier value, and its contribution, back down to a sane
    range comparable to the rest of the distribution.
    """
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(
        rng.normal(loc=0.5, scale=0.2, size=(n, len(FEATURE_NAMES))),
        columns=FEATURE_NAMES,
    )
    # Inject one extreme outlier into current_ratio, matching the real dataset's issue
    X.loc[0, "current_ratio"] = 500000.0
    y = pd.Series(rng.integers(0, 2, size=n) < 0.05, dtype=int)  # arbitrary sparse labels

    X_train, X_test = X.iloc[:400], X.iloc[400:]

    # --- WITHOUT winsorization: outlier survives into scaled space ---
    scaler_raw = StandardScaler()
    X_train_scaled_raw = scaler_raw.fit_transform(X_train)
    outlier_scaled_raw = X_train_scaled_raw[0][FEATURE_NAMES.index("current_ratio")]

    # --- WITH winsorization (same approach as train.py) ---
    lower = X_train.quantile(0.01)
    upper = X_train.quantile(0.99)
    X_train_clipped = X_train.clip(lower=lower, upper=upper, axis=1)
    scaler_clipped = StandardScaler()
    X_train_scaled_clipped = scaler_clipped.fit_transform(X_train_clipped)
    outlier_scaled_clipped = X_train_scaled_clipped[0][FEATURE_NAMES.index("current_ratio")]

    assert abs(outlier_scaled_raw) > 15, "Fixture should reproduce the distortion (large scaled outlier)"
    assert abs(outlier_scaled_clipped) < 5, (
        f"Winsorization should bring the scaled outlier back to a sane range, "
        f"got {outlier_scaled_clipped}"
    )


def test_rescale_bounds_cover_all_feature_names():
    from models.ml_distress.features import DOMAIN_RESCALE_BOUNDS
    assert set(DOMAIN_RESCALE_BOUNDS.keys()) == set(FEATURE_NAMES)


def test_rescale_clips_out_of_range_values():
    # Extreme values should clip to exactly 0 or 1, never go outside
    extreme_low = [-999] * len(FEATURE_NAMES)
    extreme_high = [999] * len(FEATURE_NAMES)
    assert all(v == 0.0 for v in rescale_live_features_to_uci_band(extreme_low))
    assert all(v == 1.0 for v in rescale_live_features_to_uci_band(extreme_high))


if __name__ == "__main__":
    import tempfile
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            if "tmp_path" in t.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    from pathlib import Path
                    t(Path(d))
            else:
                t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} tests passed.")
