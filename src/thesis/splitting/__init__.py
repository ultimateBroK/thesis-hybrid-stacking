"""Stage 3: train/val/test split with purge+embargo and correlation filtering."""

from .split import split_data, _apply_purge_embargo, _log_distribution  # noqa: F401
from .correlation import _drop_correlated  # noqa: F401
from thesis.constants import EXCLUDE_COLS as _EXCLUDE_COLS  # noqa: F401 (public re-export)

__all__ = ["split_data"]
