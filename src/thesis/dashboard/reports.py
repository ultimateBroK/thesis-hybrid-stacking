"""Report section: markdown reports and downloads only."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from thesis.dashboard.shared import trim_generated_visual_sections


def render_reports_section(data: dict, config: object, session_dir: str) -> None:
    """Render report markdown plus downloadable generated files."""
    st.markdown("> Home > **Report**")
    st.header("Report")

    reports_dir = Path(session_dir) / "reports"
    thesis_tab, eval_tab, download_tab = st.tabs(
        ["Thesis Report", "Model Evaluation", "Download Files"]
    )

    with thesis_tab:
        _render_markdown(reports_dir / "thesis_report.md")

    with eval_tab:
        _render_markdown(reports_dir / "model_evaluation.md")

    with download_tab:
        _render_downloads(reports_dir)


def _render_markdown(path: Path) -> None:
    if not path.exists():
        st.info(f"No {path.name} available.")
        return
    st.markdown(trim_generated_visual_sections(path.read_text()))


def _render_downloads(reports_dir: Path) -> None:
    files = [
        "thesis_report.md",
        "model_evaluation.md",
        "model_comparison.csv",
        "feature_importance.json",
        "walk_forward_history.json",
    ]
    available = [reports_dir / name for name in files if (reports_dir / name).exists()]
    if not available:
        st.info("No report files available for download.")
        return

    for path in available:
        st.download_button(
            path.name,
            data=path.read_text(),
            file_name=path.name,
            mime="text/plain",
            width="stretch",
        )


__all__ = ["render_reports_section"]
