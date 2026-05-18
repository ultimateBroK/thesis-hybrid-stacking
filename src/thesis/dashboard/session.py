"""Session discovery and data loading."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import streamlit as st


def find_sessions() -> list[Path]:
    """Find session dirs in results/ with config subdirectory."""
    results = Path("results")
    if not results.exists():
        return []

    def _parse_ts(path: Path) -> datetime | None:
        m = re.search(r"(\d{8})_(\d{6})$", path.name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None

    sessions = sorted(
        [p for p in results.iterdir() if p.is_dir() and (p / "config").exists()],
        key=lambda p: _parse_ts(p) or datetime.min,
        reverse=True,
    )
    return sessions


def parse_session_meta(name: str) -> dict[str, str]:
    """Parse session dirname → symbol, timeframe, date, time."""
    parts = name.split("_")
    if len(parts) >= 4:
        return {
            "symbol": parts[0],
            "timeframe": parts[1],
            "date": f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}",
            "time": f"{parts[3][:2]}:{parts[3][2:4]}:{parts[3][4:6]}",
        }
    return {"symbol": "?", "timeframe": "?", "date": "?", "time": "?"}


@st.cache_data(ttl=60)
def load_config(session_dir: str) -> dict:
    """Load config + chart data for session. Cached 60s."""
    from pathlib import Path

    from thesis.shared.config import load_config as _load_config
    from thesis.visualization.artifacts import (
        load_dashboard_artifacts as load_session_data,
    )

    sd = Path(session_dir)
    snapshot = sd / "config" / "config_snapshot.toml"
    config = (
        _load_config(snapshot) if snapshot.exists() else _load_config("config.toml")
    )

    # Wire session paths (same as shared.session_paths.configure_session_paths)
    config.paths.session_dir = str(sd)
    config.paths.model = str(sd / "models" / "lightgbm_model.pkl")
    config.paths.predictions = str(sd / "predictions" / "final_predictions.csv")
    config.paths.backtest_results = str(sd / "backtest" / "backtest_results.json")
    config.paths.report = str(sd / "reports" / "thesis_report.md")

    data = load_session_data(config)
    return {"config": config, "data": data}


@st.fragment(run_every="30s")
def session_selector_fragment() -> None:
    """Sidebar session picker. Auto-refreshes every interaction."""
    sessions = find_sessions()
    if not sessions:
        st.session_state.selected_session = None
        return

    session_names = [s.name for s in sessions]

    # Track known sessions; toast on new
    known = st.session_state.get("known_sessions", set())
    current_set = set(session_names)
    new_sessions = current_set - known
    if new_sessions and known:
        shown = st.session_state.get("shown_toasts", set())
        for ns in sorted(new_sessions):
            if ns not in shown:
                meta = parse_session_meta(ns)
                st.toast(f"New session: {meta['date']} {meta['time']}", icon="📈")
                shown.add(ns)
        st.session_state.shown_toasts = shown
    st.session_state.known_sessions = current_set

    # Build labels
    session_labels = []
    for name in session_names:
        meta = parse_session_meta(name)
        session_labels.append(
            f"{meta['date']} {meta['time']} ({meta['symbol']} {meta['timeframe']})"
        )

    current = st.session_state.get("selected_session")
    idx = 0
    if current in session_names:
        idx = session_names.index(current)
    else:
        st.session_state.selected_session = session_names[0]

    selected_label = st.selectbox(
        "Select session",
        options=session_labels,
        index=idx,
        # Auto-generated key to prevent DuplicateElementKey in fragment
    )
    selected = session_names[session_labels.index(selected_label)]
    st.session_state.selected_session = selected

    if st.button("Refresh", width="stretch"):
        st.rerun()

    st.caption("Run `pixi run workflow` to generate new sessions")
