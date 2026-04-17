"""Shared test fixtures."""

import sys
from functools import wraps
from pathlib import Path

import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _patch_pandas_axis_positional() -> None:
    """Allow positional axis arg in pandas DataFrame/Series min/max.

    backtesting._plotting calls df.min(1) and df.max(1) with axis as a
    positional argument. In pandas 4.0 this becomes keyword-only (axis=...).
    This patch makes min() and max() accept axis as both positional and keyword,
    routing it correctly regardless of how the caller passes it.

    The patch is applied before any test imports backtesting so all calls
    from the library (and our own code) are handled uniformly.
    """

    def _wrap_axis_method(name: str, cls: type) -> None:
        original = getattr(cls, name)
        assert callable(original), f"{cls.__name__}.{name} is not callable"

        @wraps(original)
        def wrapper(obj, axis=0, *args, **kwargs):
            if args and not kwargs.get("axis"):
                # Positional axis arg present — accept it as the first positional
                axis = args[0]
                args = args[1:]
            return original(obj, axis=axis, *args, **kwargs)

        setattr(cls, name, wrapper)

    for cls in (pd.DataFrame, pd.Series):
        for method in ("min", "max"):
            if hasattr(cls, method):
                _wrap_axis_method(method, cls)


_patch_pandas_axis_positional()
