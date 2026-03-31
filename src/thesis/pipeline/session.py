"""Session management for thesis pipeline workflow.

Handles session folder creation, config snapshots, and artifact organization.
Each workflow run gets its own isolated session folder for reproducibility.

Session folder structure:
    results/{SYMBOL}_{TIMEFRAME}_{YYYYMMDD}_{HHMMSS}/
    ├── config/
    │   └── config_snapshot.toml          # Copy of config used for this run
    │   └── session_info.json             # Run metadata
    ├── models/
    │   ├── lightgbm_model.pkl
    │   ├── lstm_model.pt
    ├── predictions/
    │   ├── lightgbm_oof.parquet
    │   ├── lstm_oof.parquet
    │   └── final_predictions.parquet
    ├── reports/
    │   ├── thesis_report.md
    │   └── shap_summary.png
    ├── backtest/
    │   └── backtest_results.json
    └── logs/
        └── pipeline_YYYYMMDD_HHMMSS.log
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from thesis.config.loader import Config
from thesis.config.loader import initialize_session

logger = logging.getLogger("thesis.session")


class SessionManager:
    """Manages a single workflow session with isolated outputs."""

    def __init__(self, config: Config, custom_session_id: str | None = None):
        """Initialize session manager.

        Args:
            config: Configuration object
            custom_session_id: Optional custom session name
        """
        self.config = config
        self.session_path = initialize_session(config, custom_session_id)
        self._original_config = None

        logger.info(f"Session initialized: {self.session_path.name}")
        logger.info(f"  Full path: {self.session_path.absolute()}")

    def create_config_snapshot(self) -> Path:
        """Copy current config and capture run metadata.

        Returns:
            Path to config snapshot file
        """
        config_path = Path(self.session_path / "config" / "config_snapshot.toml")

        # Copy current config.toml
        if Path("config.toml").exists():
            import shutil

            shutil.copy("config.toml", config_path)
            logger.info(f"  Config snapshot: {config_path}")

        # Create session info JSON
        session_info = {
            "session_id": self.config.paths.session_id,
            "session_path": str(self.session_path),
            "created_at": datetime.now().isoformat(),
            "symbol": self.config.data.symbol,
            "timeframe": self.config.data.timeframe,
            "git_commit": _get_git_commit(),
            "python_version": sys.version,
            "platform": sys.platform,
        }

        info_path = self.session_path / "config" / "session_info.json"
        with open(info_path, "w") as f:
            json.dump(session_info, f, indent=2)

        logger.info(f"  Session info: {info_path}")
        return config_path

    def update_config_paths(self, config: Config) -> Config:
        """Redirect all artifact paths to session folder.

        Updates model, prediction, and report paths to be session-relative.
        Global data paths (features, labels) remain unchanged.

        Args:
            config: Configuration to update

        Returns:
            Updated configuration
        """
        if not config.paths.use_sessions:
            return config

        # Store original config for reference
        self._original_config = config

        session_base = str(self.session_path)

        # Update model paths
        config.models["tree"].model_path = f"{session_base}/models/lightgbm_model.pkl"
        config.models[
            "tree"
        ].predictions_path = f"{session_base}/predictions/lightgbm_oof.parquet"

        config.models["lstm"].model_path = f"{session_base}/models/lstm_model.pt"
        config.models[
            "lstm"
        ].predictions_path = f"{session_base}/predictions/lstm_oof.parquet"

        config.models[
            "stacking"
        ].model_path = f"{session_base}/models/stacking_meta_learner.pkl"
        config.models[
            "stacking"
        ].meta_predictions_path = (
            f"{session_base}/predictions/stacking_predictions.parquet"
        )

        # Update backtest path
        config.backtest.backtest_results_path = (
            f"{session_base}/backtest/backtest_results.json"
        )

        # Update reporting paths
        config.reporting.report_path = f"{session_base}/reports/thesis_report.md"
        config.reporting.report_json_path = f"{session_base}/reports/thesis_report.json"
        config.reporting.shap_summary_path = f"{session_base}/reports/shap_summary.png"
        config.reporting.feature_importance_path = (
            f"{session_base}/reports/feature_importance.png"
        )

        # Update paths references
        config.paths.final_predictions = (
            f"{session_base}/predictions/final_predictions.parquet"
        )
        config.paths.backtest_results = f"{session_base}/backtest/backtest_results.json"
        config.paths.final_report = f"{session_base}/reports/thesis_report.md"

        logger.info("  Config paths updated for session")
        return config

    def get_log_path(self) -> Path:
        """Get path for session log file.

        Returns:
            Path to log file in session/logs/
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.session_path / "logs" / f"pipeline_{timestamp}.log"


def _get_git_commit() -> str | None:
    """Get current git commit hash if available.

    Returns:
        Commit hash string or None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def list_sessions(
    symbol: str | None = None, timeframe: str | None = None
) -> list[dict]:
    """List all available sessions with metadata.

    Args:
        symbol: Filter by symbol (optional)
        timeframe: Filter by timeframe (optional)

    Returns:
        List of session metadata dictionaries
    """
    results_dir = Path("results")
    if not results_dir.exists():
        return []

    sessions = []

    for item in results_dir.iterdir():
        if not item.is_dir():
            continue

        name = item.name

        # Skip 'latest' symlink
        if name == "latest":
            continue

        # Parse session name: SYMBOL_TIMEFRAME_YYYYMMDD_HHMMSS
        parts = name.split("_")
        if len(parts) < 4:
            continue

        session_symbol = parts[0]
        session_timeframe = parts[1]

        # Apply filters
        if symbol and session_symbol.upper() != symbol.upper():
            continue
        if timeframe and session_timeframe.upper() != timeframe.upper():
            continue

        # Load metadata if available
        info_path = item / "config" / "session_info.json"
        metadata = {}
        if info_path.exists():
            with open(info_path) as f:
                metadata = json.load(f)

        sessions.append(
            {
                "session_id": name,
                "path": str(item),
                "symbol": session_symbol,
                "timeframe": session_timeframe,
                "created_at": metadata.get("created_at", "Unknown"),
                "git_commit": metadata.get("git_commit"),
            }
        )

    # Sort by creation time (newest first)
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return sessions


def get_session_path(session_id: str) -> Path | None:
    """Get full path for a specific session.

    Args:
        session_id: Session identifier (folder name)

    Returns:
        Path to session folder or None if not found
    """
    session_path = Path("results") / session_id
    if session_path.exists() and session_path.is_dir():
        return session_path
    return None
