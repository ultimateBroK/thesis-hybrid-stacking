"""Interactive Streamlit dashboard for thesis visualization.

Launch: ``pixi run streamlit``

Orchestrates session selection, configuration display, and navigation
between the five dashboard sections. All section rendering is delegated
to ``sections/`` submodules.
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure src/ is on path for imports
_src = str(Path(__file__).resolve().parent.parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from thesis.dashboard.sections.backtest import _render_backtest_section  # noqa: E402
from thesis.dashboard.sections.data import _render_data_section  # noqa: E402
from thesis.dashboard.sections.model import _render_model_section  # noqa: E402
from thesis.dashboard.sections.reports import _render_reports_section  # noqa: E402
from thesis.dashboard.sections.training import _render_training_section  # noqa: E402
from thesis.dashboard.sessions import (  # noqa: E402
    _load_config,
    _session_selector_fragment,
)

logger = logging.getLogger("thesis.app_streamlit")


def main() -> None:
    """Render the Streamlit dashboard with session selection and navigation.

    Sets up page layout and styling, discovers and loads a selected session
    from the local results directory, and dispatches rendering to the
    appropriate section renderer.
    """
    st.set_page_config(
        page_title="Thesis Dashboard — XAU/USD",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
        /* AMOLED glass effect for metric cards */
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
        /* Subtle glow on hover */
        .stMetric:hover {
            border-color: rgba(255,255,255,0.15);
            box-shadow: 0 4px 30px rgba(0,0,0,0.4),
                        inset 0 1px 0 rgba(255,255,255,0.1),
                        0 0 20px rgba(37,99,235,0.05);
            transition: all 0.2s ease;
        }
        /* Compact sidebar spacing */
        .stSidebar .stExpander details summary {
            font-weight: 600;
            font-size: 0.9rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── Sidebar Header ──
    st.sidebar.markdown("### 📈 Thesis Dashboard")
    st.sidebar.caption("Hybrid GRU + LightGBM — XAU/USD")

    # ── Session Selector ──
    with st.sidebar.expander("📁 Session", expanded=True):
        selected = _session_selector_fragment()

    if selected is None:
        st.error("No session results found. Run `pixi run workflow` first.")
        return

    # ── Navigation ──
    sections = ["📊 Data", "🧠 Model", "🏃 Training", "💰 Backtest", "📝 Reports"]
    section_map = {
        "📊 Data": "Data Exploration",
        "🧠 Model": "Model Performance",
        "🏃 Training": "Training",
        "💰 Backtest": "Backtest Results",
        "📝 Reports": "Reports",
    }

    current_section = st.session_state.get("nav_section", "📊 Data")

    left_spacer, nav_center, right_spacer = st.columns([0.2, 0.6, 0.2])
    with nav_center:
        nav_cols = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
        for i, sec in enumerate(sections):
            with nav_cols[i]:
                btn_type = "primary" if sec == current_section else "secondary"
                if st.button(
                    sec, key=f"nav_{sec}", type=btn_type, use_container_width=True
                ):
                    st.session_state.nav_section = sec
                    st.rerun()

    section = st.session_state.get("nav_section", "📊 Data")

    # ── Load data ──
    session_path = str(Path("results") / selected)
    loaded = _load_config(session_path)
    config = loaded["config"]
    data = loaded["data"]
    metrics = data.get("metrics", {})

    # ── Configuration sidebar ──
    with st.sidebar.expander("⚙️ Configuration", expanded=False):
        st.markdown(
            f"**GRU**: hidden={config.gru.hidden_size}, layers={config.gru.num_layers}, "
            f"seq={config.gru.sequence_length}, epochs={config.gru.epochs}"
        )
        st.markdown(
            f"**LightGBM**: leaves={config.model.num_leaves}, "
            f"depth={config.model.max_depth}, lr={config.model.learning_rate}"
        )
        st.markdown(
            f"**Backtest**: leverage={config.backtest.leverage}:1, "
            f"lots={config.backtest.lots_per_trade}, "
            f"conf≥{config.backtest.confidence_threshold}"
        )
        st.markdown(
            f"**Split**: train={config.splitting.train_start[:10]}→"
            f"{config.splitting.train_end[:10]}"
        )

    # ── Quick Stats sidebar ──
    if metrics:
        with st.sidebar.expander("📊 Quick Stats", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Return", f"{metrics.get('return_pct', 0):.2f}%")
            c2.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.2f}%")
            c1.metric("Trades", f"{metrics.get('num_trades', 0)}")
            c2.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")

    # ── Render selected section ──
    section_name = section_map[section]
    if section_name == "Data Exploration":
        _render_data_section(data, config)
    elif section_name == "Model Performance":
        _render_model_section(data, session_path)
    elif section_name == "Training":
        _render_training_section(data, session_path)
    elif section_name == "Reports":
        _render_reports_section(session_path)
    else:
        _render_backtest_section(data)


if __name__ == "__main__":
    main()
