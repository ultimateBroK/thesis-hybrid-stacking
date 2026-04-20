"""Session discovery and configuration loading for the dashboard.

Handles finding session directories, parsing session names, loading
config snapshots, and rendering the Streamlit session selector fragment.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from thesis.charts import load_session_data
from thesis.config import load_config

logger = logging.getLogger("thesis.app_streamlit")


def _find_sessions() -> list[Path]:
    """Discover available session directories under the local results/ folder.

    A session directory is any immediate subdirectory that contains a
    ``config`` entry. Returns sessions sorted by timestamp descending.

    Returns:
        Reverse-sorted list of session directory paths; empty if none found.
    """
    results = Path("results")
    if not results.exists():
        return []

    def parse_session_timestamp(path: Path) -> datetime | None:
        """Extract and parse timestamp from session directory name.

        Session directories follow the pattern:
        ``{SYMBOL}_{TIMEFRAME}_{YYYYMMDD}_{HHMMSS}``

        Returns:
            datetime if parsing succeeds, None otherwise.
        """
        m = re.search(r"(\d{8})_(\d{6})$", path.name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None

    sessions = sorted(
        [p for p in results.iterdir() if p.is_dir() and (p / "config").exists()],
        key=lambda p: parse_session_timestamp(p) or datetime.min,
        reverse=True,
    )
    return sessions


def _parse_session_meta(name: str) -> dict[str, str]:
    """Parse a session directory name into its metadata fields.

    Args:
        name: Session directory name, expected as
            ``SYMBOL_TIMEFRAME_YYYYMMDD_HHMMSS``.

    Returns:
        Mapping with keys ``symbol``, ``timeframe``, ``date``, ``time``.
        Values default to ``"?"`` if parsing fails.
    """
    parts = name.split("_")
    if len(parts) >= 4:
        return {
            "symbol": parts[0],
            "timeframe": parts[1],
            "date": f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}",
            "time": f"{parts[3][:2]}:{parts[3][2:4]}:{parts[3][4:6]}",
        }
    return {"symbol": "?", "timeframe": "?", "date": "?", "time": "?"}


@st.cache_resource(ttl=60)
def _load_config(session_dir: str) -> dict:
    """Load configuration and session data for a given session directory.

    Cached for a short duration to reduce I/O overhead.

    Args:
        session_dir: Path to the session directory.

    Returns:
        Mapping with keys ``config`` (Config object) and ``data`` (session data).
    """
    config = load_config()
    config.paths.session_dir = session_dir
    snapshot = Path(session_dir) / "config" / "config_snapshot.toml"
    if snapshot.exists():
        config = load_config(snapshot)
        config.paths.session_dir = session_dir
    data = load_session_data(config)
    return {"config": config, "data": data}


@st.fragment(run_every=30)
def _session_selector_fragment() -> str | None:
    """Render a sidebar session selector and return the chosen session name.

    Updates Streamlit session state keys ``known_sessions`` and
    ``selected_session``. Shows toasts for newly discovered sessions.

    Returns:
        The raw session directory name, or None if no sessions are available.
    """
    sessions = _find_sessions()
    if not sessions:
        return None

    session_names = [s.name for s in sessions]

    known = st.session_state.get("known_sessions", set())
    current_set = set(session_names)
    new_sessions = current_set - known
    if new_sessions and known:
        for ns in sorted(new_sessions):
            meta = _parse_session_meta(ns)
            st.toast(f"🆕 New session: {meta['date']} {meta['time']}", icon="📈")
    st.session_state.known_sessions = current_set

    session_labels = []
    for name in session_names:
        meta = _parse_session_meta(name)
        session_labels.append(
            f"{meta['date']} {meta['time']} ({meta['symbol']} {meta['timeframe']})"
        )

    current = st.session_state.get("selected_session")
    if current in session_names:
        idx = session_names.index(current)
    else:
        idx = 0
        st.session_state.selected_session = session_names[0]

    selected_label = st.selectbox(
        "Select session",
        options=session_labels,
        index=idx,
        key="_session_selectbox",
    )
    selected = session_names[session_labels.index(selected_label)]
    st.session_state.selected_session = selected

    if st.button("🔄 Refresh", width="stretch", key="_refresh_btn"):
        st.rerun()

    st.caption("Run `pixi run workflow` to generate new sessions")
    return selected
