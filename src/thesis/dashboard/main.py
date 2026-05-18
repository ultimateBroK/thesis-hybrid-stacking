"""Dashboard: session selection, navigation, section dispatch."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from thesis.dashboard.backtest import render_backtest_section
from thesis.dashboard.data import render_data_section
from thesis.dashboard.model import render_evaluation_section, render_model_section
from thesis.dashboard.reports import render_reports_section
from thesis.dashboard.session import load_config, session_selector_fragment
from thesis.dashboard.shared import render_config_summary
from thesis.visualization.summaries import compute_prediction_summary

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
    "📦 Dataset": ("Dataset", render_data_section),
    "🤖 Model": ("Model", render_model_section),
    "📊 Evaluation": ("Evaluation", render_evaluation_section),
    "📄 Report": ("Report", render_reports_section),
    "💼 Demo": ("Demo", render_backtest_section),
}


def configure_dashboard_page() -> None:
    """Set page config, CSS, sidebar header."""
    st.set_page_config(
        page_title="ML Result Viewer — XAU/USD Hybrid Stacking",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    st.sidebar.markdown("### ML Result Viewer")
    st.sidebar.caption("XAU/USD Hybrid Stacking")


def select_session() -> str | None:
    """Render session selector, return chosen session name."""
    with st.sidebar.expander("Session", expanded=True):
        session_selector_fragment()
        return st.session_state.get("selected_session")


def show_missing_session_message() -> None:
    """Show error when no session exists."""
    st.error("No session results found. Run `pixi run workflow` first.")


def select_dashboard_section() -> str:
    """Render nav buttons, return selected section key."""
    sections = list(_SECTION_RENDERERS)
    current = st.session_state.get("nav_section", "📦 Dataset")
    if current not in _SECTION_RENDERERS:
        current = "📦 Dataset"
        st.session_state.nav_section = current

    _, nav_center, _ = st.columns([0.2, 0.6, 0.2])
    with nav_center:
        nav_cols = st.columns(len(sections))
        for i, sec in enumerate(sections):
            with nav_cols[i]:
                btn_type = "primary" if sec == current else "secondary"
                if st.button(sec, key=f"nav_{sec}", type=btn_type, width="stretch"):
                    st.session_state.nav_section = sec
                    st.rerun()

    section = st.session_state.get("nav_section", "📦 Dataset")
    if section not in _SECTION_RENDERERS:
        section = "📦 Dataset"
    return section


def load_dashboard_session(selected_session: str) -> dict:
    """Load config + artifacts for a session."""
    session_path = str(Path("results") / selected_session)
    loaded = load_config(session_path)
    return {
        "config": loaded["config"],
        "data": loaded["data"],
        "session_path": session_path,
    }


def render_dashboard_sidebar(session: dict) -> None:
    """Sidebar: config summary + ML quick stats."""
    config = session["config"]
    data = session["data"]

    with st.sidebar.expander("Configuration", expanded=False):
        render_config_summary(config)

    ml_metrics = compute_prediction_summary(data)
    if ml_metrics:
        with st.sidebar.expander("ML Quick Stats", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{ml_metrics['accuracy']:.1%}")
            c2.metric("Macro F1", f"{ml_metrics['macro_f1']:.3f}")
            c1.metric("Directional", f"{ml_metrics['directional_accuracy']:.1%}")
            c2.metric("Predictions", f"{ml_metrics['total_predictions']:,}")


def render_dashboard_section(section: str, session: dict) -> None:
    """Dispatch to the section renderer."""
    _, renderer = _SECTION_RENDERERS[section]
    renderer(session["data"], session["config"], session["session_path"])


def main() -> None:
    """Render dashboard: sidebar, nav, load session, dispatch section."""
    configure_dashboard_page()
    selected_session = select_session()

    if selected_session is None:
        show_missing_session_message()
        return

    section = select_dashboard_section()
    session = load_dashboard_session(selected_session)
    render_dashboard_sidebar(session)
    render_dashboard_section(section, session)


if __name__ == "__main__":
    main()
