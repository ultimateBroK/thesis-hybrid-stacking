"""Column contract validation for pipeline stage boundaries."""

from __future__ import annotations

import polars as pl


def _check_columns(df: pl.DataFrame, required: set[str], schema_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{schema_name}: missing columns {sorted(missing)}. "
            f"Got {sorted(df.columns)}"
        )


class OhlcvSchema:
    """Validate raw OHLCV column contract."""

    _COLS: frozenset[str] = frozenset(
        ["timestamp", "open", "high", "low", "close", "volume"]
    )

    @classmethod
    def validate(cls, df: pl.DataFrame, config: object | None = None) -> None:
        """Raise ValueError if required OHLCV columns are missing."""
        _check_columns(df, cls._COLS, cls.__name__)


class FeaturesSchema:
    """Validate feature-enriched column contract."""

    _STRUCTURAL: frozenset[str] = frozenset(
        ["timestamp", "open", "high", "low", "close", "volume"]
    )

    @classmethod
    def validate(cls, df: pl.DataFrame, config: object | None = None) -> None:
        """Raise ValueError if structural or full feature columns are missing."""
        _check_columns(df, cls._STRUCTURAL, cls.__name__)
        if config is not None:
            from thesis.shared.feature_registry import build_feature_output_cols

            expected = set(build_feature_output_cols(config))
            _check_columns(df, expected, f"{cls.__name__}(full)")


class LabelsSchema:
    """Validate labelled column contract."""

    _STRUCTURAL: frozenset[str] = frozenset(
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "label",
            "upper_barrier",
            "lower_barrier",
            "touched_bar",
            "event_end",
            "sample_weight",
        ]
    )

    @classmethod
    def validate(cls, df: pl.DataFrame, config: object | None = None) -> None:
        """Raise ValueError if label metadata or feature columns are missing."""
        _check_columns(df, cls._STRUCTURAL, cls.__name__)
        if config is not None:
            from thesis.shared.feature_registry import build_feature_output_cols

            feature_cols = set(build_feature_output_cols(config))
            _check_columns(df, feature_cols, f"{cls.__name__}(full)")
