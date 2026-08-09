"""
Quant Forge — Premium Dashboard
================================
Streamlit multi-panel analytics interface.
Dark theme, Plotly interactive charts, real-time refresh.

Layout:
  [Header + Controls]
  [1: Live Candlestick] [2: Correlation Heatmap]
  [3: Risk Metrics]     [4: Monte Carlo Paths]
  [5: Efficient Frontier (Classical + Quantum)]
  [6: GARCH Volatility + Kalman Smoothing]
"""

import sys
import os
import time
import logging
import warnings
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Adjust path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NSE_TICKERS, US_TICKERS, TRADING_DAYS
from data.pipeline import DataFetcher, FeatureEngineer, load_universe, is_market_open
from models.risk_engine import (
    GBM,
    MonteCarloEngine,
    RiskMetrics,
    PerformanceMetrics,
    CAPM,
    GARCH,
    KalmanFilter,
    PCAFactorModel,
    BayesianVaR,
    compute_full_risk_report,
)
from models.optimizer import (
    ClassicalOptimizer,
    QuantumOptimizer,
    compute_all_portfolios,
    buy_and_hold_benchmark,
    equal_weight_portfolio,
)

warnings.filterwarnings("ignore")
log = logging.getLogger("quant_forge.app")

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="⚛️ Quant Forge | NSE Risk Engine",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Quant Forge — Quantum-Inspired Financial Analytics. Built by Rajnish Singh.",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark Theme + Quant Forge Branding
# ─────────────────────────────────────────────────────────────────────────────

DARK_CSS = """
<style>
    /* ── Core Dark Theme ── */
    .stApp { background: #0a0e1a; color: #e8eaf6; font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 1rem 2rem; max-width: 1800px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1f2937; }
    [data-testid="stSidebar"] .stMarkdown { color: #9ca3af; }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label { color: #9ca3af !important; font-size: 12px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffd700 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* ── Headers ── */
    h1 { background: linear-gradient(135deg, #ffd700, #00b4d8);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-size: 2.5rem !important; font-weight: 800; }
    h2 { color: #ffd700; font-weight: 700; border-bottom: 1px solid #1f2937; padding-bottom: 8px; }
    h3 { color: #00b4d8; font-weight: 600; }

    /* ── Dataframes ── */
    .dataframe { background: #111827 !important; color: #e8eaf6 !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #ffd700, #f59e0b);
        color: #0a0e1a;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(255,215,0,0.3); }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 11px;
        color: #9ca3af;
        margin: 2px;
    }
    .badge-gold { background: rgba(255,215,0,0.15); border-color: #ffd700; color: #ffd700; }
    .badge-blue { background: rgba(0,180,216,0.15); border-color: #00b4d8; color: #00b4d8; }

    /* ── Status ── */
    .status-live { color: #10b981; font-weight: 600; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #1f2937; margin: 1rem 0; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { color: #9ca3af; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background: #1f2937; color: #ffd700; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly Template
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#0a0e1a",
    font=dict(family="Inter", color="#e8eaf6", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚛️ Quant Forge")
    st.markdown(
        '<span class="badge badge-gold">v2.0 Research</span> <span class="badge badge-blue">NSE Live</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Asset selection
    st.markdown("### 📊 Universe")
    selected_universe = st.radio(
        "Market",
        ["NSE India", "US Equities", "Mixed", "Crypto"],
        index=0,
        horizontal=True,
    )
    if selected_universe == "NSE India":
        universe = NSE_TICKERS
    elif selected_universe == "US Equities":
        universe = US_TICKERS
    elif selected_universe == "Mixed":
        universe = NSE_TICKERS[:10] + US_TICKERS[:5]
    else:
        universe = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"]

    selected_tickers = st.multiselect(
        "Assets",
        universe,
        default=universe[:8] if len(universe) >= 8 else universe,
    )
    if len(selected_tickers) < 2:
        st.warning("Select ≥ 2 assets")
        st.stop()

    st.markdown("### ⚙️ Parameters")
    risk_aversion = st.slider("Risk Aversion λ", 0.1, 5.0, 1.0, 0.1)
    lookback_period = st.selectbox("Lookback", ["6mo", "1y", "2y", "3y"], index=1)
    mc_paths = st.slider("MC Paths (K)", 1, 50, 10) * 1000
    var_conf = st.selectbox("VaR Confidence", [0.90, 0.95, 0.99], index=1)
    include_quantum = st.checkbox("⚛️ Quantum Optimization", value=True)
    n_quantum = st.slider("Quantum Assets", 4, 10, 6)

    st.markdown("### 🔄 Refresh")
    auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
    st.markdown(f"**NSE Market:** {market_status}")
    st.caption(
        f"Last update: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%H:%M:%S IST')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading (Cached)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=300, show_spinner=False)
def load_data(tickers, period):
    return load_universe(tickers, period=period)


@st.cache_data(ttl=60, show_spinner=False)
def load_live_quotes(tickers):
    quotes = {}
    for t in tickers:
        q = DataFetcher.get_quote(t)
        if q:
            quotes[t] = q
    return quotes


@st.cache_data(ttl=600, show_spinner=False)
def run_optimization(returns_json, include_q, n_q):
    returns = pd.read_json(returns_json)
    results = {}
    classical = ClassicalOptimizer(returns)
    results["max_sharpe"] = classical.max_sharpe()
    results["min_vol"] = classical.min_volatility()
    results["risk_parity"] = classical.risk_parity()
    if include_q:
        q_ret = returns.iloc[:, : min(n_q, len(returns.columns))]
        quantum = QuantumOptimizer(q_ret, n_select=min(5, n_q), p=2)
        results["quantum"] = quantum.optimize()
    return results, classical.compute_frontier(30)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# ⚛️ Quant Forge | NSE Risk & Quantum Portfolio Engine")
st.markdown(
    '<span class="badge badge-gold">PhD-Grade Quant Research</span> '
    '<span class="badge badge-blue">Live NSE Data</span> '
    '<span class="badge">GBM + GARCH + Kalman</span> '
    '<span class="badge">QAOA Quantum Opt</span> '
    '<span class="badge">Monte Carlo 10K</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("🔄 Fetching live data..."):
    try:
        prices, returns, features = load_data(selected_tickers, lookback_period)
        quotes = load_live_quotes(selected_tickers)
    except Exception as e:
        st.error(f"Data load error: {e}")
        st.code(traceback.format_exc())
        st.stop()

# Align to common tickers
common = [t for t in selected_tickers if t in prices.columns]
if not common:
    st.error("No data found for selected tickers.")
    st.stop()
prices = prices[common]
returns = returns[common]

# ─────────────────────────────────────────────────────────────────────────────
# Live Quote Bar
# ─────────────────────────────────────────────────────────────────────────────

if quotes:
    cols = st.columns(min(len(quotes), 8))
    for i, (ticker, q) in enumerate(list(quotes.items())[:8]):
        pct = q.get("pct", 0)
        with cols[i % len(cols)]:
            delta_color = "normal" if pct >= 0 else "inverse"
            st.metric(
                label=ticker.replace(".NS", ""),
                value=(
                    f"₹{q['price']:,.1f}" if ".NS" in ticker else f"${q['price']:,.2f}"
                ),
                delta=f"{pct:+.2f}%",
                delta_color=delta_color,
            )
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Equal-weight baseline
# ─────────────────────────────────────────────────────────────────────────────
n_assets = len(common)
ew_weights = np.ones(n_assets) / n_assets

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 Price & Technicals",
        "🎲 Monte Carlo Risk",
        "⚡ Efficient Frontier",
        "⚛️ Quantum Optimizer",
        "📊 Risk Dashboard",
        "🔬 Factor Analysis",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Price & Technicals
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## 📈 Price Action & Technical Indicators")

    sel_ticker = st.selectbox("Focus Asset", common, index=0)

    col1, col2 = st.columns([3, 1])

    with col1:
        # ── Candlestick ──────────────────────────────────────────────────────
        try:
            ticker_obj = __import__("yfinance").download(
                sel_ticker,
                period=lookback_period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if len(ticker_obj) > 0:
                fig_candle = go.Figure(
                    data=[
                        go.Candlestick(
                            x=ticker_obj.index,
                            open=ticker_obj["Open"],
                            high=ticker_obj["High"],
                            low=ticker_obj["Low"],
                            close=ticker_obj["Close"],
                            name=sel_ticker,
                            increasing_line_color="#10b981",
                            decreasing_line_color="#ef4444",
                        )
                    ]
                )
                # Add Bollinger Bands
                bb_upper, bb_mid, bb_lower = FeatureEngineer.bollinger_bands(
                    ticker_obj["Close"], 20, 2.0
                )
                for band, name, color in [
                    (bb_upper, "BB Upper", "#ffd700"),
                    (bb_mid, "BB Mid", "#9ca3af"),
                    (bb_lower, "BB Lower", "#00b4d8"),
                ]:
                    fig_candle.add_trace(
                        go.Scatter(
                            x=band.index,
                            y=band.values,
                            name=name,
                            line=dict(color=color, width=1, dash="dot"),
                            opacity=0.7,
                        )
                    )

                fig_candle.update_layout(
                    **PLOTLY_THEME,
                    title=f"{sel_ticker} — OHLCV with Bollinger Bands",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=450,
                )
                st.plotly_chart(fig_candle, use_container_width=True)
        except Exception as e:
            st.warning(f"Candlestick error: {e}")
            # Fallback: line chart
            fig_line = px.line(
                prices[sel_ticker].reset_index(),
                x="Date",
                y=sel_ticker,
                title=f"{sel_ticker} Close",
            )
            fig_line.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        # ── RSI ──────────────────────────────────────────────────────────────
        rsi_series = FeatureEngineer.rsi(prices[sel_ticker])
        current_rsi = float(rsi_series.dropna().iloc[-1])
        rsi_signal = (
            "🔴 Overbought"
            if current_rsi > 70
            else ("🟢 Oversold" if current_rsi < 30 else "⚪ Neutral")
        )

        st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_signal)

        # Vol
        rets_sel = returns[sel_ticker]
        ann_vol = float(rets_sel.std() * np.sqrt(252))
        st.metric("Ann. Volatility", f"{ann_vol:.1%}")

        # Performance
        perf = PerformanceMetrics(rets_sel).summary()
        st.metric("Sharpe Ratio", f"{perf['sharpe']:.3f}")
        st.metric("Max Drawdown", f"{perf['max_drawdown']:.1%}")
        st.metric("CAGR", f"{perf['cagr']:.1%}")
        st.metric("Sortino", f"{perf['sortino']:.3f}")

    # ── Kalman Smoothing ─────────────────────────────────────────────────────
    st.markdown("### 🔬 Kalman Filter Price Smoothing")
    kf = KalmanFilter(process_noise=1e-4, obs_noise=0.5)
    kf_df = kf.smooth(prices[sel_ticker].dropna())

    fig_kalman = go.Figure()
    fig_kalman.add_trace(
        go.Scatter(
            x=kf_df.index,
            y=kf_df["price"],
            name="Raw Price",
            line=dict(color="#4b5563", width=1),
            opacity=0.5,
        )
    )
    fig_kalman.add_trace(
        go.Scatter(
            x=kf_df.index,
            y=kf_df["smoothed"],
            name="Kalman Smoothed",
            line=dict(color="#ffd700", width=2),
        )
    )
    fig_kalman.update_layout(
        **PLOTLY_THEME,
        title="Kalman Filter: Price Level + Trend Estimation",
        height=300,
        xaxis_title="Date",
        yaxis_title="Price",
    )
    st.plotly_chart(fig_kalman, use_container_width=True)

    # ── Correlation Heatmap ──────────────────────────────────────────────────
    st.markdown("### 🔗 Correlation Matrix")
    corr = returns.corr()
    tick_labels = [t.replace(".NS", "") for t in corr.columns]

    fig_corr = px.imshow(
        corr,
        x=tick_labels,
        y=tick_labels,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdYlGn",
        title="Return Correlations (Full Lookback)",
        aspect="auto",
        text_auto=".2f",
    )
    fig_corr.update_layout(**PLOTLY_THEME, height=400)
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Monte Carlo Risk
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 🎲 Monte Carlo Risk Simulation")
    st.markdown("""
    **Method**: Correlated GBM paths via Cholesky decomposition.
    `S_{t+dt} = S_t · exp[(μ - σ²/2)dt + σ√dt · ε]`
    """)

    mc_col1, mc_col2 = st.columns(2)
    with mc_col1:
        mc_horizon = st.slider("Horizon (Trading Days)", 10, 252, 63)
    with mc_col2:
        mc_display_paths = st.slider("Display Paths", 50, 500, 200)

    if st.button("▶ Run Monte Carlo Simulation"):
        with st.spinner(f"Running {mc_paths:,} paths × {mc_horizon}d..."):
            mc_engine = MonteCarloEngine(
                returns.iloc[-504:],  # 2y lookback
                ew_weights,
                n_paths=mc_paths,
                horizon=mc_horizon,
            )
            mc_result = mc_engine.run()

        # ── Metrics ──────────────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("VaR 95%", f"{mc_result.var_95:.2%}")
        m2.metric("VaR 99%", f"{mc_result.var_99:.2%}")
        m3.metric("ES 95%", f"{mc_result.es_95:.2%}")
        m4.metric("Mean P&L", f"{mc_result.mean_pnl:.2%}")
        m5.metric("P(Loss)", f"{mc_result.prob_loss:.1%}")

        # ── Path Chart ────────────────────────────────────────────────────────
        paths = mc_result.paths[:mc_display_paths, :]
        t_axis = np.arange(paths.shape[1])

        fig_paths = go.Figure()
        for i in range(min(mc_display_paths, 100)):
            fig_paths.add_trace(
                go.Scatter(
                    x=t_axis,
                    y=paths[i],
                    line=dict(color="#00b4d8", width=0.3),
                    opacity=0.15,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        # Percentiles
        for pct, color, name in [
            (5, "#ef4444", "5th Pct"),
            (50, "#ffd700", "Median"),
            (95, "#10b981", "95th Pct"),
        ]:
            fig_paths.add_trace(
                go.Scatter(
                    x=t_axis,
                    y=np.percentile(mc_result.paths, pct, axis=0),
                    name=name,
                    line=dict(color=color, width=2),
                )
            )
        fig_paths.update_layout(
            **PLOTLY_THEME,
            title=f"Monte Carlo Paths: {mc_paths:,} simulations, {mc_horizon}d horizon",
            xaxis_title="Trading Days",
            yaxis_title="Relative Price",
            height=400,
        )
        st.plotly_chart(fig_paths, use_container_width=True)

        # ── PnL Histogram ─────────────────────────────────────────────────────
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=mc_result.pnl * 100,
                nbinsx=80,
                marker_color="#00b4d8",
                opacity=0.7,
                name="P&L (%)",
            )
        )
        fig_hist.add_vline(
            x=-mc_result.var_95 * 100,
            line_color="#ffd700",
            line_dash="dash",
            annotation_text="VaR 95%",
        )
        fig_hist.add_vline(
            x=-mc_result.var_99 * 100,
            line_color="#ef4444",
            line_dash="dash",
            annotation_text="VaR 99%",
        )
        fig_hist.update_layout(
            **PLOTLY_THEME,
            title="P&L Distribution (Monte Carlo)",
            xaxis_title="P&L (%)",
            yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("👆 Click **Run Monte Carlo Simulation** to start")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Efficient Frontier
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## ⚡ Efficient Frontier — Classical vs Quantum")
    st.markdown("""
    Classical frontier via convex Markowitz optimization.
    Quantum: QAOA on QUBO with cardinality constraint.
    """)

    with st.spinner("Computing frontiers..."):
        try:
            classical_opt = ClassicalOptimizer(returns)
            frontier = classical_opt.compute_frontier(30)

            ms_result = classical_opt.max_sharpe()
            mv_result = classical_opt.min_volatility()
            rp_result = classical_opt.risk_parity()

            if include_quantum:
                q_returns = returns.iloc[:, : min(n_quantum, n_assets)]
                quantum = QuantumOptimizer(q_returns, n_select=min(5, n_quantum), p=2)
                q_result = quantum.optimize()
                q_frontier = quantum.quantum_frontier(8)

        except Exception as e:
            st.error(f"Optimization error: {e}\n{traceback.format_exc()}")
            st.stop()

    # ── Frontier Plot ─────────────────────────────────────────────────────────
    fig_ef = go.Figure()

    # Classical frontier
    if frontier:
        ef_vols = [p.vol for p in frontier]
        ef_rets = [p.ret for p in frontier]
        ef_sharpes = [p.sharpe for p in frontier]
        fig_ef.add_trace(
            go.Scatter(
                x=ef_vols,
                y=ef_rets,
                mode="lines",
                line=dict(color="#00b4d8", width=3),
                name="Classical Frontier",
            )
        )

    # Key portfolios
    for label, res, color, sym in [
        ("Max Sharpe", ms_result, "#ffd700", "star"),
        ("Min Vol", mv_result, "#10b981", "diamond"),
        ("Risk Parity", rp_result, "#f59e0b", "circle"),
    ]:
        fig_ef.add_trace(
            go.Scatter(
                x=[res.expected_vol],
                y=[res.expected_ret],
                mode="markers+text",
                marker=dict(
                    color=color, size=14, symbol=sym, line=dict(color="white", width=2)
                ),
                text=[label],
                textposition="top center",
                name=label,
            )
        )

    # Quantum portfolio
    if include_quantum and q_result:
        fig_ef.add_trace(
            go.Scatter(
                x=[q_result.expected_vol],
                y=[q_result.expected_ret],
                mode="markers+text",
                marker=dict(
                    color="#a855f7",
                    size=18,
                    symbol="hexagram",
                    line=dict(color="white", width=2),
                ),
                text=["⚛️ Quantum"],
                textposition="top center",
                name="Quantum QAOA",
            )
        )
        if q_frontier:
            fig_ef.add_trace(
                go.Scatter(
                    x=[p.vol for p in q_frontier],
                    y=[p.ret for p in q_frontier],
                    mode="lines",
                    line=dict(color="#a855f7", width=2, dash="dash"),
                    name="Quantum Frontier",
                    opacity=0.7,
                )
            )

    # Equal weight
    ew_r = float(ew_weights @ returns.mean().values * 252)
    ew_v = float(np.sqrt(ew_weights @ (returns.cov().values * 252) @ ew_weights))
    fig_ef.add_trace(
        go.Scatter(
            x=[ew_v],
            y=[ew_r],
            mode="markers+text",
            marker=dict(color="#6b7280", size=10, symbol="cross"),
            text=["Equal Weight"],
            textposition="top center",
            name="Equal Weight",
        )
    )

    fig_ef.update_layout(
        **PLOTLY_THEME,
        title="Efficient Frontier: Classical Markowitz vs Quantum QAOA",
        xaxis_title="Expected Volatility (Annualized)",
        yaxis_title="Expected Return (Annualized)",
        height=550,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # ── Comparison Table ──────────────────────────────────────────────────────
    st.markdown("### Portfolio Performance Comparison")
    comparison_data = []
    for label, res in [
        ("Max Sharpe", ms_result),
        ("Min Vol", mv_result),
        ("Risk Parity", rp_result),
    ]:
        comparison_data.append(
            {
                "Strategy": label,
                "Exp. Return": f"{res.expected_ret:.1%}",
                "Exp. Vol": f"{res.expected_vol:.1%}",
                "Sharpe": f"{res.sharpe:.3f}",
                "Method": res.method,
            }
        )
    if include_quantum and q_result:
        comparison_data.append(
            {
                "Strategy": "⚛️ Quantum QAOA",
                "Exp. Return": f"{q_result.expected_ret:.1%}",
                "Exp. Vol": f"{q_result.expected_vol:.1%}",
                "Sharpe": f"{q_result.sharpe:.3f}",
                "Method": "QAOA/QUBO",
            }
        )
    st.dataframe(
        pd.DataFrame(comparison_data), use_container_width=True, hide_index=True
    )

    # ── Weight Allocation Chart ───────────────────────────────────────────────
    st.markdown("### Portfolio Weight Allocation")
    w_col1, w_col2 = st.columns(2)
    for col, (label, result) in zip(
        [w_col1, w_col2], [("Max Sharpe", ms_result), ("Min Vol", mv_result)]
    ):
        w_dict = {
            k.replace(".NS", ""): v for k, v in result.weights.items() if v > 0.001
        }
        fig_pie = px.pie(
            values=list(w_dict.values()),
            names=list(w_dict.keys()),
            title=f"{label} Weights",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Plasma_r,
        )
        fig_pie.update_layout(**PLOTLY_THEME, height=320)
        col.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Quantum Optimizer
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## ⚛️ Quantum Portfolio Optimization")
    st.markdown("""
    **QUBO Formulation**: Asset selection via binary optimization.
    
    $$H_Q = \\lambda_{\\text{risk}} \\mathbf{w}^T \\Sigma \\mathbf{w} - \\lambda_{\\text{ret}} \\boldsymbol{\\mu}^T \\mathbf{w} + \\lambda_c\\left(\\sum_i x_i - k\\right)^2$$
    
    Solved via **QAOA** (Quantum Approximate Optimization Algorithm) on Qiskit Aer simulator.
    """)

    if not include_quantum:
        st.info("Enable **Quantum Optimization** in sidebar to run.")
    else:
        q_returns_sub = returns.iloc[:, : min(n_quantum, n_assets)]
        q_opt = QuantumOptimizer(q_returns_sub, n_select=min(5, n_quantum), p=2)

        if st.button("⚛️ Run QAOA Optimization"):
            with st.spinner("Running QAOA on Qiskit Aer..."):
                q_res = q_opt.optimize()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Exp. Return", f"{q_res.expected_ret:.1%}")
            c2.metric("Exp. Vol", f"{q_res.expected_vol:.1%}")
            c3.metric("Sharpe", f"{q_res.sharpe:.3f}")
            c4.metric("Assets Selected", q_res.metadata.get("n_selected", 0))

            st.markdown("### Selected Portfolio")
            selected = q_res.metadata.get("selected_assets", [])
            if selected:
                st.success(
                    f"**Quantum-Selected Assets:** {', '.join([s.replace('.NS','') for s in selected])}"
                )

            # Weight bar chart
            w_df = pd.DataFrame(
                [
                    {"Asset": k.replace(".NS", ""), "Weight": v}
                    for k, v in q_res.weights.items()
                    if v > 0.001
                ]
            ).sort_values("Weight", ascending=True)

            fig_bar = px.bar(
                w_df,
                x="Weight",
                y="Asset",
                orientation="h",
                color="Weight",
                color_continuous_scale="Plasma",
                title="Quantum QAOA Portfolio Weights",
            )
            fig_bar.update_layout(**PLOTLY_THEME, height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

            # QUBO matrix visualization
            st.markdown("### QUBO Matrix Heatmap")
            Q = q_opt._build_qubo()
            q_labels = [t.replace(".NS", "") for t in q_returns_sub.columns]
            fig_qubo = px.imshow(
                Q,
                x=q_labels,
                y=q_labels,
                color_continuous_scale="RdYlBu_r",
                title="QUBO Matrix Q_{ij}",
                text_auto=".2f",
            )
            fig_qubo.update_layout(**PLOTLY_THEME, height=400)
            st.plotly_chart(fig_qubo, use_container_width=True)

            # Quantum circuit info
            st.markdown("### Quantum Circuit Summary")
            st.code(
                f"""
QAOA Circuit Configuration:
  - Qubits:     {min(n_quantum, n_assets)} (one per asset)
  - QAOA Depth: p = {q_opt.p}
  - Parameters: 2p = {2*q_opt.p} variational parameters
  - Gates:      {min(n_quantum,n_assets)} H gates (initial |+⟩)
                + p × [U_C(γ) + U_B(β)] layers
  - Backend:    Qiskit Aer Statevector Simulator
  - Shots:      4096
  - Optimizer:  COBYLA (classical outer loop)
  
Problem:
  - n = {min(n_quantum, n_assets)} assets → 2^{min(n_quantum, n_assets)} = {2**min(n_quantum, n_assets)} states
  - k = {min(5, n_quantum)} assets selected (cardinality constraint)
  - QUBO energy at solution: {q_res.metadata.get('qubo_energy', 'N/A'):.4f}
            """,
                language="text",
            )
        else:
            st.info("👆 Click **Run QAOA Optimization** to start quantum solver")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Risk Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## 📊 Comprehensive Risk Dashboard")

    # ── CAPM ─────────────────────────────────────────────────────────────────
    st.markdown("### 📐 CAPM Analysis (Nifty50 Benchmark)")
    with st.spinner("Running CAPM..."):
        try:
            nifty_prices = DataFetcher.get_nifty50()
            nifty_returns = nifty_prices.pct_change().dropna()

            # Align
            idx = returns.index.intersection(nifty_returns.index)
            capm_model = CAPM(returns.loc[idx], nifty_returns.loc[idx])
            capm_res = capm_model.fit()

            capm_df = pd.DataFrame(capm_res).T.reset_index()
            capm_df.columns = [
                "Asset",
                "Alpha (Ann.)",
                "Beta",
                "R²",
                "Exp. Return",
                "Treynor",
            ]
            capm_df["Asset"] = capm_df["Asset"].str.replace(".NS", "")
            st.dataframe(capm_df, use_container_width=True, hide_index=True)

            # Beta vs Alpha scatter
            fig_capm = px.scatter(
                capm_df,
                x="Beta",
                y="Alpha (Ann.)",
                text="Asset",
                size_max=14,
                color="R²",
                color_continuous_scale="Viridis",
                title="CAPM: Alpha vs Beta (Jensen's Alpha)",
                labels={"Alpha (Ann.)": "α (Annualized)"},
            )
            fig_capm.add_vline(
                x=1,
                line_dash="dash",
                line_color="#9ca3af",
                annotation_text="Market Beta=1",
            )
            fig_capm.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
            fig_capm.update_traces(textposition="top center")
            fig_capm.update_layout(**PLOTLY_THEME, height=400)
            st.plotly_chart(fig_capm, use_container_width=True)
        except Exception as e:
            st.warning(f"CAPM error: {e}")

    # ── GARCH Volatility ─────────────────────────────────────────────────────
    st.markdown("### 📉 GARCH(1,1) Conditional Volatility")
    garch_ticker = st.selectbox("Asset for GARCH", common, key="garch_sel")

    with st.spinner(f"Fitting GARCH(1,1) for {garch_ticker}..."):
        try:
            garch_model = GARCH(returns[garch_ticker].dropna())
            garch_model.fit()
            cond_vol = garch_model.conditional_vol()
            params = garch_model.params()

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("ω (omega)", f"{params['omega']:.6f}")
            g2.metric("α (arch)", f"{params['alpha']:.4f}")
            g3.metric("β (garch)", f"{params['beta']:.4f}")
            g4.metric("Persistence α+β", f"{params['persistence']:.4f}")

            fig_garch = go.Figure()
            fig_garch.add_trace(
                go.Scatter(
                    x=cond_vol.index,
                    y=cond_vol.values,
                    fill="tozeroy",
                    fillcolor="rgba(0,180,216,0.1)",
                    line=dict(color="#00b4d8", width=2),
                    name="GARCH(1,1) Conditional Vol",
                )
            )
            fig_garch.add_trace(
                go.Scatter(
                    x=returns[garch_ticker].rolling(21).std().dropna().index
                    * np.sqrt(252),
                    y=returns[garch_ticker].rolling(21).std().dropna() * np.sqrt(252),
                    line=dict(color="#ffd700", width=1, dash="dot"),
                    name="Rolling 21d Vol",
                )
            )
            fig_garch.update_layout(
                **PLOTLY_THEME,
                title=f"GARCH(1,1): {garch_ticker} Conditional Volatility",
                xaxis_title="Date",
                yaxis_title="Ann. Volatility",
                height=380,
            )
            st.plotly_chart(fig_garch, use_container_width=True)
        except Exception as e:
            st.warning(f"GARCH error: {e}")

    # ── VaR Comparison ────────────────────────────────────────────────────────
    st.markdown("### 🔴 VaR / ES — All Methods")
    var_ticker = st.selectbox("Asset for VaR", common, key="var_sel")
    rm = RiskMetrics(returns[var_ticker], confidence=var_conf)
    var_res = rm.all_methods()
    bayes_var = BayesianVaR(returns[var_ticker]).posterior_var()

    var_rows = []
    for method, vals in var_res.items():
        var_rows.append(
            {
                "Method": method.replace("_", " ").title(),
                f"VaR {var_conf:.0%}": f"{vals['var']:.4f} ({vals['var']*100:.2f}%)",
                f"ES {var_conf:.0%}": f"{vals['es']:.4f} ({vals['es']*100:.2f}%)",
            }
        )
    var_rows.append(
        {
            "Method": "Bayesian (NIG Prior)",
            f"VaR {var_conf:.0%}": f"{bayes_var['posterior_var']:.4f}",
            f"ES {var_conf:.0%}": f"{bayes_var['posterior_es']:.4f}",
        }
    )
    st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

    # ── Drawdown Chart ────────────────────────────────────────────────────────
    st.markdown("### 📉 Drawdown Analysis")
    port_price = (prices * ew_weights).sum(axis=1)
    cumret = port_price / port_price.iloc[0]
    rolling_max = cumret.cummax()
    drawdown = (cumret - rolling_max) / rolling_max

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.2)",
            line=dict(color="#ef4444", width=1),
            name="Portfolio Drawdown",
        )
    )
    fig_dd.update_layout(
        **PLOTLY_THEME,
        title="Portfolio Underwater Equity Curve",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        height=300,
        yaxis_tickformat=".1%",
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: Factor Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("## 🔬 PCA Factor Analysis & Multi-Param Heatmap")

    with st.spinner("Running PCA..."):
        pca_model = PCAFactorModel(n_components=min(5, n_assets - 1))
        pca_model.fit(returns)
        pca_summary = pca_model.explained_variance_summary()
        factor_rets = pca_model.factor_returns(returns)

    # Scree plot
    fig_scree = go.Figure()
    fig_scree.add_trace(
        go.Bar(
            x=pca_summary.index,
            y=pca_summary["var_ratio"],
            marker_color="#00b4d8",
            name="Var Ratio",
        )
    )
    fig_scree.add_trace(
        go.Scatter(
            x=pca_summary.index,
            y=pca_summary["cumulative"],
            line=dict(color="#ffd700", width=2),
            name="Cumulative",
            yaxis="y2",
        )
    )
    fig_scree.update_layout(
        **PLOTLY_THEME,
        title="PCA Scree Plot — Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Variance Ratio",
        yaxis2=dict(title="Cumulative", overlaying="y", side="right", tickformat=".0%"),
        height=350,
    )
    st.plotly_chart(fig_scree, use_container_width=True)

    # Factor loadings heatmap
    loadings = pd.DataFrame(
        pca_model.eigvecs,
        columns=returns.columns,
        index=[f"PC{i+1}" for i in range(len(pca_model.eigvecs))],
    )
    loadings.columns = [c.replace(".NS", "") for c in loadings.columns]

    fig_load = px.imshow(
        loadings,
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="PCA Factor Loadings (Eigenvectors)",
        text_auto=".2f",
    )
    fig_load.update_layout(**PLOTLY_THEME, height=350)
    st.plotly_chart(fig_load, use_container_width=True)

    # ── Multi-Param Risk/Return Heatmap ──────────────────────────────────────
    st.markdown("### 🌡️ Multi-Parameter Risk–Return Heatmap")
    st.markdown(
        "Sweeping **Risk Aversion (λ)** × **Lookback Window** → Sharpe surface."
    )

    lambdas = np.linspace(0.5, 5.0, 8)
    windows = [42, 63, 126, 252]
    sharpe_grid = np.zeros((len(lambdas), len(windows)))

    with st.spinner("Computing multi-param grid..."):
        for i, lam in enumerate(lambdas):
            for j, win in enumerate(windows):
                try:
                    r_sub = returns.iloc[-win:]
                    if len(r_sub) < 20:
                        continue
                    mu_ = r_sub.mean().values * 252
                    cov_ = r_sub.cov().values * 252
                    # MV with risk aversion
                    from scipy.optimize import minimize as sp_min

                    def obj(w):
                        return lam * w @ cov_ @ w - w @ mu_

                    w0 = np.ones(n_assets) / n_assets
                    res = sp_min(
                        obj,
                        w0,
                        bounds=[(0, 1)] * n_assets,
                        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                        method="SLSQP",
                    )
                    w = np.maximum(res.x, 0)
                    w /= w.sum()
                    r_ = w @ mu_
                    v_ = np.sqrt(w @ cov_ @ w)
                    sharpe_grid[i, j] = (r_ - 0.07) / (v_ + 1e-12)
                except:
                    pass

    fig_heatmap = px.imshow(
        sharpe_grid,
        x=[f"{w}d" for w in windows],
        y=[f"λ={l:.1f}" for l in lambdas],
        color_continuous_scale="Plasma",
        title="Sharpe Ratio: Risk Aversion (λ) × Lookback Window",
        text_auto=".2f",
        aspect="auto",
    )
    fig_heatmap.update_layout(**PLOTLY_THEME, height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Factor Return Time Series ─────────────────────────────────────────────
    st.markdown("### 📈 Principal Factor Returns")
    fig_factors = go.Figure()
    colors_f = ["#ffd700", "#00b4d8", "#10b981", "#f59e0b", "#a855f7"]
    for i, col in enumerate(factor_rets.columns):
        cumr = (1 + factor_rets[col]).cumprod()
        fig_factors.add_trace(
            go.Scatter(
                x=cumr.index,
                y=cumr.values,
                name=col,
                line=dict(color=colors_f[i % 5], width=2),
            )
        )
    fig_factors.update_layout(
        **PLOTLY_THEME,
        title="Cumulative Factor Returns (PCA Components)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        height=350,
    )
    st.plotly_chart(fig_factors, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#4b5563; font-size:12px;">
    ⚛️ <strong>Quant Forge</strong> — Built by <em>Rajnish Singh</em> | 
    NSE India + Quantum ML | 
    <span style="color:#ffd700">GBM · GARCH · Kalman · CAPM · QAOA</span> |
    arXiv-ready research prototype
    </div>
    """,
    unsafe_allow_html=True,
)

# Auto-refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()
