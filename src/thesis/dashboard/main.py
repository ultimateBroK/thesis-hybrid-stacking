"""Dashboard: session selection, navigation, section dispatch."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from thesis.dashboard.backtest import render_backtest_section
from thesis.dashboard.data import render_data_section
from thesis.dashboard.model import render_model_section
from thesis.dashboard.reports import render_reports_section
from thesis.dashboard.session import load_config, session_selector_fragment
from thesis.dashboard.shared import render_config_summary
from thesis.dashboard.training import render_training_section

# AMOLED glass — metric cards, hover glow
_CSS = """
<style>
    .stMetric {
        background: linear-gradient(135deg,
            rgba(255,255,255,0.05) 0%,
            rgba(255,255,255,0.02) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3),
                    inset 0 1px 0 rgba(255,255,255,0.06);
    }
    .stMetric label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.6);
        letter-spacing: 0.02em;
    }
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .stMetric div[data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }
    .stMetric:hover {
        border-color: rgba(255,255,255,0.15);
        box-shadow: 0 4px 30px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(255,255,255,0.1),
                    0 0 20px rgba(37,99,235,0.05);
        transition: all 0.2s ease;
    }
    .stSidebar .stExpander details summary {
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
"""

_SECTION_RENDERERS: dict[str, tuple[str, object]] = {
    "📊 Data": ("Data Exploration", render_data_section),
    "🧠 Model": ("Model Performance", render_model_section),
    "🏃 Training": ("Training", render_training_section),
    "💰 Backtest": ("Backtest Results", render_backtest_section),
    "📝 Reports": ("Reports", render_reports_section),
}


def main() -> None:
    """Render dashboard: sidebar, nav, load session, dispatch section."""
    st.set_page_config(
        page_title="Thesis Dashboard — XAU/USD",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CSS, unsafe_allow_html=True)

    # Sidebar header
    st.sidebar.markdown("### 📈 Thesis Dashboard")
    st.sidebar.caption("Hybrid Stacking — XAU/USD")

    # Session selector (auto-refreshes every 30s via @st.fragment)
    with st.sidebar.expander("Session", expanded=True):
        session_selector_fragment()
        selected = st.session_state.get("selected_session")

    if selected is None:
        st.error("No session results found. Run `pixi run workflow` first.")
        return

    # Nav: 5 buttons, current section highlighted
    sections = list(_SECTION_RENDERERS)
    current_section = st.session_state.get("nav_section", "📊 Data")

    _, nav_center, _ = st.columns([0.2, 0.6, 0.2])
    with nav_center:
        nav_cols = st.columns(5)
        for i, sec in enumerate(sections):
            with nav_cols[i]:
                btn_type = "primary" if sec == current_section else "secondary"
                if st.button(
                    sec, key=f"nav_{sec}", type=btn_type, width='stretch'
                ):
                    st.session_state.nav_section = sec
                    st.rerun()

    section = st.session_state.get("nav_section", "📊 Data")

    # Load session data
    session_path = str(Path("results") / selected)
    loaded = load_config(session_path)
    config = loaded["config"]
    data = loaded["data"]
    metrics = data.get("metrics", {})

    # Sidebar: config summary + quick stats
    with st.sidebar.expander("Configuration", expanded=False):
        render_config_summary(config)

    if metrics:
        with st.sidebar.expander("Quick Stats", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Return", f"{metrics.get('return_pct', 0):.2f}%")
            c2.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.2f}%")
            c1.metric("Trades", f"{metrics.get('num_trades', 0)}")
            c2.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")

    # Dispatch section renderer
    _name, renderer = _SECTION_RENDERERS[section]
    if section == "💰 Backtest":
        renderer(data, config, session_path)
    else:
        renderer(data, session_path)


main()
