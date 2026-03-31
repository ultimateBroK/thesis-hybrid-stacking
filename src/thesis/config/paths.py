"""Path utilities for thesis data management."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    """Project path configuration with methods for symbol-specific paths."""

    base_raw_dir: Path = Path("data/raw")
    base_state_dir: Path = Path("data/state")

    def raw_data_dir(self, symbol: str) -> Path:
        """Get raw data directory for a specific symbol."""
        path = self.base_raw_dir / symbol
        path.mkdir(parents=True, exist_ok=True)
        return path

    def state_file(self, symbol: str) -> Path:
        """Get state file path for a specific symbol."""
        self.base_state_dir.mkdir(parents=True, exist_ok=True)
        return self.base_state_dir / f"{symbol}_download_state.json"


# Default paths instance
DEFAULT_PATHS = ProjectPaths()
