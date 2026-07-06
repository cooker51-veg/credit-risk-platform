"""
app.py — Credit Risk & Distress Prediction Platform (Phase 5: Dashboard UI)

Run locally with:
    streamlit run app.py

Integrates:
  - Phase 1: data_pipeline.fetch_financials (5-year statement pull via yfinance)
  - Phase 2: models.credit_ratios (Altman Z/Z''/Ohlson O-Score, liquidity/leverage/profitability)
  - Phase 3: models.ml_distress (logistic regression distress model, trained locally
    via `python -m models.ml_distress.train` — this app loads the saved artifacts,
    it does not retrain)
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_pipeline.fetch_financials import fetch_company_financials, FinancialDataError
from models.credit_ratios import (
    build_annual_inputs, altman_z_score, altman_z_double_prime_score,
    ohlson_o_score, liquidity_ratios, leverage_ratios, profitability_trend,
)

# ML artifacts are optional — app must not crash if the user hasn't run
# `python -m models.ml_distress.train` locally yet.
try:
    import joblib
    from pathlib import Path
    from models.ml_distress.features import FEATURE_NAMES, features_from_annual_inputs, rescale_live_features_to_uci_band
    ML_ARTIFACT_DIR = Path(__file__).parent / "models" / "ml_distress" / "artifacts"
    ML_AVAILABLE = (ML_ARTIFACT_DIR / "logreg_model.joblib").exists()
except ImportError:
    ML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page config + design tokens
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Credit Risk & Distress Prediction Platform",
                    page_icon="\u25C8", layout="wide", initial_sidebar_state="expanded")

COLORS = {
    "bg": "#0B1220", "panel": "#131B2E", "panel_border": "#243049",
    "text": "#E5E7EB", "muted": "#94A3B8",
    "safe": "#10B981", "grey": "#F59E0B", "distress": "#EF4444",
    "accent": "#5EEAD4",
}

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
  .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
  h1, h2, h3 {{ font-family: 'Newsreader', serif !important; font-weight: 600 !important; color: {COLORS['text']}; }}
  [data-testid="stSidebar"] {{ background-color: {COLORS['panel']}; border-right: 1px solid {COLORS['panel_border']}; }}
  .metric-card {{
      background-color: {COLORS['panel']}; border: 1px solid {COLORS['panel_border']};
      border-left: 4px solid var(--zone-color, {COLORS['accent']});
      border-radius: 6px; padding: 16px 18px; margin-bottom: 10px;
  }}
  .metric-label {{ font-size: 0.78rem; color: {COLORS['muted']}; text-transform: uppercase; letter-spacing: 0.04em; }}
  .metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 500; margin-top: 4px; }}
  .metric-zone {{ font-size: 0.85rem; margin-top: 2px; }}
  .formula-box {{
      background-color: {COLORS['panel']}; border: 1px solid {COLORS['panel_border']};
      border-radius: 6px; padding: 12px 16px; font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem; color: {COLORS['muted']}; margin-bottom: 12px;
  }}
  .disclaimer {{
      background-color: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3);
      border-radius: 6px; padding: 12px 16px; font-size: 0.85rem; color: {COLORS['text']};
  }}
  .stTabs [data-baseweb="tab"] {{ font-family: 'Inter', sans-serif; }}
</style>
""", unsafe_allow_html=True)

ZONE_COLOR = {"Safe": COLORS["safe"], "Grey": COLORS["grey"], "Distress": COLORS["distress"],
              "Low distress probability": COLORS["safe"], "Elevated risk": COLORS["grey"],
              "High distress probability": COLORS["distress"], "N/A": COLORS["muted"]}


def metric_card(label, value, zone=None, sub=None):
    color = ZONE_COLOR.get(zone, COLORS["accent"])
    zone_html = f'<div class="metric-zone" style="color:{color}">{zone}</div>' if zone else ""
    sub_html = f'<div class="metric-zone" style="color:{COLORS["muted"]}">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card" style="--zone-color:{color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {zone_html}{sub_html}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data + computation
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_company(ticker: str):
    financials = fetch_company_financials(ticker)
    annual_inputs = build_annual_inputs(
        financials.income_statement_tidy, financials.balance_sheet_tidy, financials.cash_flow_tidy
    )
    return financials, annual_inputs


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_cap(ticker: str):
    import yfinance as yf
    try:
        return yf.Ticker(ticker).info.get("marketCap")
    except Exception:
        return None


def compute_all_ratios(annual_inputs: pd.DataFrame, market_cap):
    years = list(annual_inputs.index)
    zpp_by_year = {y: altman_z_double_prime_score(annual_inputs.loc[y]) for y in years}
    latest = years[-1]
    z_latest = altman_z_score(annual_inputs.loc[latest], market_cap) if market_cap else None
    o_by_year = {}
    for i, y in enumerate(years):
        if i == 0:
            continue
        o_by_year[y] = ohlson_o_score(annual_inputs.loc[y], annual_inputs.loc[years[i - 1]])
    liq = liquidity_ratios(annual_inputs.loc[latest])
    lev = leverage_ratios(annual_inputs.loc[latest])
    trend = profitability_trend(annual_inputs)
    return {"years": years, "zpp": zpp_by_year, "z_latest": z_latest, "o": o_by_year,
            "liq": liq, "lev": lev, "trend": trend, "latest_year": latest}


def predict_ml_distress(annual_inputs: pd.DataFrame):
    if not ML_AVAILABLE:
        return None
    model = joblib.load(ML_ARTIFACT_DIR / "logreg_model.joblib")
    scaler = joblib.load(ML_ARTIFACT_DIR / "scaler.joblib")
    results = {}
    for year, row in annual_inputs.iterrows():
        feats = features_from_annual_inputs(row)
        if feats is None:
            continue
        rescaled = rescale_live_features_to_uci_band(feats)
        X = np.array(rescaled).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        contributions = model.coef_[0] * X_scaled[0]
        results[year] = {"probability": prob, "features": dict(zip(FEATURE_NAMES, feats)),
                          "contributions": dict(zip(FEATURE_NAMES, contributions))}
    return results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("### \u25C8 Credit Risk Platform")
st.sidebar.caption("Enter an NSE (.NS), BSE (.BO), or US ticker.")
main_ticker = st.sidebar.text_input("Company ticker", value="RELIANCE.NS")

st.sidebar.markdown("---")
st.sidebar.markdown("**Peer comparison** (optional)")
peer1 = st.sidebar.text_input("Peer 1 ticker", value="")
peer2 = st.sidebar.text_input("Peer 2 ticker", value="")

analyze = st.sidebar.button("Analyze", type="primary", use_container_width=True)

if not ML_AVAILABLE:
    st.sidebar.markdown("""
    <div class="disclaimer" style="font-size:0.78rem; margin-top:16px;">
    ML distress model not found. Run <code>python -m models.ml_distress.train</code>
    locally once to enable that tab.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Credit Risk & Distress Prediction Platform")
st.caption("Analytical / educational tool — not validated for real lending or investment decisions.")

if not main_ticker:
    st.info("Enter a ticker in the sidebar and click Analyze to begin.")
    st.stop()

if analyze or "last_ticker" in st.session_state:
    st.session_state["last_ticker"] = main_ticker

    try:
        with st.spinner(f"Pulling 5-year financials for {main_ticker}..."):
            financials, annual_inputs = load_company(main_ticker)
    except FinancialDataError as e:
        st.error(str(e))
        st.stop()

    if not annual_inputs.index.tolist():
        st.error("No usable fiscal years found for this ticker.")
        st.stop()

    market_cap = get_market_cap(main_ticker)
    ratios = compute_all_ratios(annual_inputs, market_cap)
    years = ratios["years"]
    latest_year = ratios["latest_year"]

    st.markdown(f"## {financials.company_name}")
    st.caption(f"{financials.ticker} \u2022 {financials.sector or 'N/A'} / {financials.industry or 'N/A'} "
               f"\u2022 Currency: {financials.currency}")

    if financials.warnings:
        with st.expander("Data warnings"):
            for w in financials.warnings:
                st.warning(w)

    tab_overview, tab_ratios, tab_ml, tab_peers, tab_methodology = st.tabs(
        ["Overview", "Credit Ratios", "ML Distress Model", "Peer Comparison", "Methodology"]
    )

    # =======================================================================
    # OVERVIEW TAB
    # =======================================================================
    with tab_overview:
        zpp_latest = ratios["zpp"][latest_year]
        o_latest = ratios["o"].get(years[-1]) if len(years) > 1 else None

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if zpp_latest["score"] is not None:
                metric_card("Altman Z''-Score (latest FY)", f"{zpp_latest['score']:.2f}", zpp_latest["zone"])
            else:
                metric_card("Altman Z''-Score", "N/A", "N/A")
        with col2:
            if o_latest and o_latest["probability"] is not None:
                metric_card("Ohlson O-Score distress prob.", f"{o_latest['probability']*100:.1f}%", o_latest["zone"])
            else:
                metric_card("Ohlson O-Score", "N/A", "N/A")
        with col3:
            cr = ratios["liq"]["current_ratio"]["value"]
            metric_card("Current Ratio", f"{cr:.2f}" if cr is not None else "N/A")
        with col4:
            de = ratios["lev"]["debt_to_equity"]["value"]
            metric_card("Debt / Equity", f"{de:.2f}" if de is not None else "N/A")

        st.markdown("#### 5-Year Trend: Revenue, EBITDA, Net Margin")
        rev = annual_inputs["total_revenue"]
        ebitda = annual_inputs["ebitda"]
        margin = (annual_inputs["net_income"] / annual_inputs["total_revenue"] * 100)

        fig = go.Figure()
        fig.add_bar(x=[y.year for y in rev.index], y=rev.values, name="Revenue",
                    marker_color=COLORS["accent"], opacity=0.85)
        fig.add_bar(x=[y.year for y in ebitda.index], y=ebitda.values, name="EBITDA",
                    marker_color="#3B82F6", opacity=0.85)
        fig.add_trace(go.Scatter(x=[y.year for y in margin.index], y=margin.values, name="Net Margin %",
                                  yaxis="y2", mode="lines+markers", line=dict(color=COLORS["grey"], width=2)))
        fig.update_layout(
            barmode="group", template="plotly_dark",
            paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
            yaxis=dict(title="Amount"), yaxis2=dict(title="Net Margin %", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1), height=380, margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Altman Z''-Score Trend (risk zone bands)")
        zpp_vals = {y: ratios["zpp"][y]["score"] for y in years if ratios["zpp"][y]["score"] is not None}
        if zpp_vals:
            yrs_x = [y.year for y in zpp_vals.keys()]
            fig2 = go.Figure()
            fig2.add_hrect(y0=-10, y1=1.1, fillcolor=COLORS["distress"], opacity=0.12, line_width=0)
            fig2.add_hrect(y0=1.1, y1=2.6, fillcolor=COLORS["grey"], opacity=0.12, line_width=0)
            fig2.add_hrect(y0=2.6, y1=15, fillcolor=COLORS["safe"], opacity=0.12, line_width=0)
            fig2.add_trace(go.Scatter(x=yrs_x, y=list(zpp_vals.values()), mode="lines+markers",
                                       line=dict(color=COLORS["accent"], width=3), marker=dict(size=9),
                                       name="Z''-Score"))
            fig2.update_layout(template="plotly_dark", paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
                                height=320, margin=dict(t=20, b=20), showlegend=False,
                                yaxis_title="Z''-Score", xaxis_title="Fiscal Year")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Green = Safe (Z''>2.6) \u2022 Amber = Grey zone (1.1\u20132.6) \u2022 Red = Distress (<1.1). "
                       "Uses book equity so it's valid across all historical years (see Methodology).")

    # =======================================================================
    # CREDIT RATIOS TAB
    # =======================================================================
    with tab_ratios:
        st.markdown("#### Altman Z''-Score \u2014 by fiscal year")
        for y in years:
            r = ratios["zpp"][y]
            with st.expander(f"FY {y.date()}: Z'' = {r['score'] if r['score'] is not None else 'N/A'} "
                              f"({r.get('zone', 'N/A')})"):
                if r["score"] is not None:
                    st.markdown(f'<div class="formula-box">{r["formula"]}</div>', unsafe_allow_html=True)
                    st.json(r["inputs"])
                    st.json(r["components"])
                else:
                    st.write(r.get("note", "Insufficient data."))

        if ratios["z_latest"] is not None:
            st.markdown("#### Altman Z-Score (classic, most recent FY only \u2014 uses current market cap)")
            r = ratios["z_latest"]
            if r["score"] is not None:
                st.markdown(f'<div class="formula-box">{r["formula"]}</div>', unsafe_allow_html=True)
                st.json(r["inputs"])
                st.caption(r["note"])
            else:
                st.write(r.get("note"))

        st.markdown("#### Ohlson O-Score \u2014 by fiscal year")
        for y, r in ratios["o"].items():
            with st.expander(f"FY {y.date()}: P(distress) = "
                              f"{r['probability']*100:.1f}% " if r.get("probability") is not None else f"FY {y.date()}: N/A"):
                if r.get("score") is not None:
                    st.markdown(f'<div class="formula-box">{r["formula"]}</div>', unsafe_allow_html=True)
                    st.json(r["inputs"])
                    st.caption(r["note"])
                else:
                    st.write(r.get("note"))

        st.markdown("#### Liquidity Ratios (latest FY)")
        for name, d in ratios["liq"].items():
            st.write(f"**{name.replace('_', ' ').title()}**: {d['value']}  \n"
                     f"*{d['formula']}* \u2014 {d['inputs']}")

        st.markdown("#### Leverage Ratios (latest FY)")
        for name, d in ratios["lev"].items():
            st.write(f"**{name.replace('_', ' ').title()}**: {d['value']}  \n"
                     f"*{d['formula']}* \u2014 {d['inputs']}")

        st.markdown("#### Profitability Trend")
        st.write(ratios["trend"].get("trend"))
        st.json({str(k.date()): v for k, v in ratios["trend"]["margins_by_year"].items()})

    # =======================================================================
    # ML DISTRESS MODEL TAB
    # =======================================================================
    with tab_ml:
        if not ML_AVAILABLE:
            st.info("ML model artifacts not found. Run `python -m models.ml_distress.train` "
                     "locally once, then reload this app.")
        else:
            ml_results = predict_ml_distress(annual_inputs)
            if not ml_results:
                st.warning("Could not compute the ML distress score for this company "
                           "(insufficient underlying data).")
            else:
                latest_ml_year = max(ml_results.keys())
                prob = ml_results[latest_ml_year]["probability"]

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    number={"suffix": "%", "font": {"color": COLORS["text"]}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                        "bar": {"color": COLORS["accent"]},
                        "steps": [
                            {"range": [0, 20], "color": COLORS["safe"]},
                            {"range": [20, 50], "color": COLORS["grey"]},
                            {"range": [50, 100], "color": COLORS["distress"]},
                        ],
                    },
                    title={"text": f"Distress Probability \u2014 FY {latest_ml_year.date()}",
                           "font": {"color": COLORS["text"], "size": 16}},
                ))
                fig_gauge.update_layout(paper_bgcolor=COLORS["panel"], height=320, margin=dict(t=60, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("#### Feature contributions (why this score)")
                contribs = ml_results[latest_ml_year]["contributions"]
                sorted_feats = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))
                fig_bar = go.Figure(go.Bar(
                    x=[v for _, v in sorted_feats], y=[k.replace("_", " ") for k, _ in sorted_feats],
                    orientation="h",
                    marker_color=[COLORS["distress"] if v > 0 else COLORS["safe"] for _, v in sorted_feats],
                ))
                fig_bar.update_layout(template="plotly_dark", paper_bgcolor=COLORS["panel"],
                                       plot_bgcolor=COLORS["panel"], height=350, margin=dict(t=20, b=20),
                                       xaxis_title="Contribution (red = toward distress, green = toward safety)")
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("""
                <div class="disclaimer">
                <b>Model limitations:</b> trained on the UCI Taiwanese Bankruptcy Prediction dataset
                (1999\u20132009, ~6,800 companies, ~3.2% bankrupt). This is a directional distress signal
                from pattern-matching on 10 standard ratios \u2014 <b>not</b> a probability calibrated to
                real-world Indian default rates. Live company ratios are rescaled using domain-typical
                bounds before scoring, since the original dataset's normalization parameters were never
                published (see Methodology tab).
                </div>
                """, unsafe_allow_html=True)

    # =======================================================================
    # PEER COMPARISON TAB
    # =======================================================================
    with tab_peers:
        peer_tickers = [t.strip() for t in [peer1, peer2] if t.strip()]
        if not peer_tickers:
            st.info("Add 1\u20132 peer tickers in the sidebar to compare.")
        else:
            all_tickers = [main_ticker] + peer_tickers
            peer_data = {}
            for t in all_tickers:
                try:
                    with st.spinner(f"Loading {t}..."):
                        f, ai = load_company(t)
                    if not ai.index.tolist():
                        continue
                    ly = ai.index[-1]
                    zpp = altman_z_double_prime_score(ai.loc[ly])
                    liq = liquidity_ratios(ai.loc[ly])
                    lev = leverage_ratios(ai.loc[ly])
                    peer_data[t] = {
                        "name": f.company_name,
                        "z_score": zpp["score"] if zpp["score"] is not None else 0,
                        "current_ratio": liq["current_ratio"]["value"] or 0,
                        "interest_coverage": lev["interest_coverage"]["value"] or 0,
                        "debt_to_equity": lev["debt_to_equity"]["value"] or 0,
                    }
                except FinancialDataError as e:
                    st.warning(f"{t}: {e}")

            if len(peer_data) >= 2:
                metrics = ["z_score", "current_ratio", "interest_coverage", "debt_to_equity"]
                labels = ["Z''-Score", "Current Ratio", "Interest Coverage", "Debt/Equity (inverted)"]

                # Normalize each axis 0-1 across the compared set for a fair radar comparison.
                # Debt/Equity is inverted (lower = better) so higher on the chart always means "stronger."
                norm_data = {t: [] for t in peer_data}
                for m in metrics:
                    vals = [peer_data[t][m] for t in peer_data]
                    lo, hi = min(vals), max(vals)
                    for t in peer_data:
                        v = peer_data[t][m]
                        n = (v - lo) / (hi - lo) if hi != lo else 0.5
                        if m == "debt_to_equity":
                            n = 1 - n
                        norm_data[t].append(n)

                fig_radar = go.Figure()
                for t, vals in norm_data.items():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=labels + [labels[0]],
                        fill="toself", name=peer_data[t]["name"],
                    ))
                fig_radar.update_layout(
                    template="plotly_dark", paper_bgcolor=COLORS["panel"],
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=450, showlegend=True,
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.caption("Each axis normalized 0\u20131 across the compared companies (relative comparison, "
                           "not absolute). Outer edge = strongest in this peer set on that metric.")

                st.markdown("#### Side-by-side")
                df_display = pd.DataFrame({t: {
                    "Company": d["name"], "Z''-Score": round(d["z_score"], 2),
                    "Current Ratio": round(d["current_ratio"], 2),
                    "Interest Coverage": round(d["interest_coverage"], 2),
                    "Debt/Equity": round(d["debt_to_equity"], 2),
                } for t, d in peer_data.items()}).T
                st.dataframe(df_display, use_container_width=True)
            else:
                st.warning("Need at least 2 valid companies to compare.")

    # =======================================================================
    # METHODOLOGY TAB
    # =======================================================================
    with tab_methodology:
        st.markdown("""
### Data Source
5 years of annual financial statements pulled live via **yfinance** (Yahoo Finance).
Free-tier limitation: Yahoo typically indexes ~4-5 years of annual statements;
fewer years may be returned for some tickers, and this is flagged automatically
as a data warning rather than silently truncated.

### Classical Credit Risk Models
- **Altman Z''-Score**: uses *book* value of equity, valid for every historical
  year \u2014 used for the 5-year trend chart. Zones: Safe (>2.6), Grey (1.1\u20132.6),
  Distress (<1.1).
- **Altman Z-Score (classic)**: uses *current* market capitalization, so it is
  only computed for the most recent fiscal year \u2014 historical market cap isn't
  available from the free data source, and using today's market cap for older
  years would silently misstate them.
- **Ohlson O-Score**: adapted for non-US markets by using raw ln(Total Assets)
  instead of Ohlson's original 1980 US GNP price-level deflator (no equivalent
  index applied here). "Funds from operations" is proxied by Operating Cash Flow.

### ML Distress Model
- Logistic regression (primary, fully explainable via coefficients) and gradient
  boosting (comparison), trained on the **UCI Taiwanese Bankruptcy Prediction
  dataset** (6,819 companies, Taiwan Economic Journal, 1999\u20132009, ~3.2% bankrupt).
- **Known limitation**: this dataset's published ratio columns were found to be
  pre-normalized into a tight [0,1] band by the original researchers, using
  unpublished internal min/max values. A live company's raw ratio looks like an
  extreme outlier against that squeezed band. Since the original normalization
  cannot be inverted, live ratios are rescaled into a comparable [0,1] band using
  standard finance-textbook typical ranges before scoring \u2014 this is an
  **approximation**, not an exact inverse transform.
- The model reflects Taiwanese companies under Taiwanese accounting standards
  from over 15 years ago \u2014 not Indian companies, not Ind-AS/IFRS, and not
  current. It captures general distress **patterns** in 10 standard ratios; it
  is **not** a probability calibrated to real-world Indian default rates.
- SHAP-style linear contributions (coefficient \u00d7 standardized value) explain
  which ratios pushed each prediction toward or away from distress.

### Peer Comparison
Radar chart axes are normalized 0\u20131 **within the compared set** \u2014 this is a
relative comparison between the specific companies selected, not an absolute
scale against the broader market or sector.

### Intended Use
This is an **analytical / educational tool**. It is not validated for, and
should not be used for, real lending, credit-rating, or investment decisions.
        """)

else:
    st.info("Enter a ticker in the sidebar and click **Analyze** to begin.")
