"""Column contract validation for pipeline stage boundaries."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


def _check_columns(df: pl.DataFrame, required: set[str], schema_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{schema_name}: missing columns {sorted(missing)}. "
            f"Got {sorted(df.columns)}"
        )


def _validate_monotonic_unique_timestamp(df: pl.DataFrame, schema_name: str) -> None:
    if "timestamp" not in df.columns or len(df) < 2:
        return
    ts = df.get_column("timestamp")
    if ts.null_count() > 0:
        raise ValueError(f"{schema_name}: timestamp contains nulls")
    if ts.n_unique() != len(ts):
        raise ValueError(f"{schema_name}: timestamp must be unique")
    deltas = ts.diff().drop_nulls().dt.total_milliseconds()
    if int((deltas <= 0).sum()) > 0:
        raise ValueError(f"{schema_name}: timestamp must be strictly increasing")


class OhlcvSchema:
    """Validate raw OHLCV column contract."""

    _COLS: frozenset[str] = frozenset(
        ["timestamp", "open", "high", "low", "close", "volume"]
    )
    _SCHEMA = pa.DataFrameSchema(
        {
            "timestamp": pa.Column(nullable=False),
            "open": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "high": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "low": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "close": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "volume": pa.Column(pl.Float64, checks=pa.Check.ge(0), coerce=True),
        },
        strict=False,
    )

    @classmethod
    def validate(cls, df: pl.DataFrame, config: object | None = None) -> None:
        """Raise ValueError if required OHLCV columns are missing."""
        _check_columns(df, cls._COLS, cls.__name__)
        cls._SCHEMA.validate(df, lazy=True)
        _validate_monotonic_unique_timestamp(df, cls.__name__)
        bad_ohlc = df.filter(
            (pl.col("high") < pl.col("low"))
            | (pl.col("open") < pl.col("low"))
            | (pl.col("open") > pl.col("high"))
            | (pl.col("close") < pl.col("low"))
            | (pl.col("close") > pl.col("high"))
        )
        if len(bad_ohlc) > 0:
            raise ValueError(f"{cls.__name__}: invalid OHLC relationships")


class FeaturesSchema:
    """Validate feature-enriched column contract."""

    _STRUCTURAL: frozenset[str] = frozenset(
        ["timestamp", "open", "high", "low", "close", "volume"]
    )
    _STRUCTURAL_SCHEMA = pa.DataFrameSchema(
        {
            "timestamp": pa.Column(nullable=False),
            "open": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "high": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "low": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "close": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "volume": pa.Column(pl.Float64, checks=pa.Check.ge(0), coerce=True),
        },
        strict=False,
    )

    @classmethod
    def validate(cls, df: pl.DataFrame, config: object | None = None) -> None:
        """Raise ValueError if structural or full feature columns are missing."""
        _check_columns(df, cls._STRUCTURAL, cls.__name__)
        cls._STRUCTURAL_SCHEMA.validate(df, lazy=True)
        _validate_monotonic_unique_timestamp(df, cls.__name__)
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
    _STRUCTURAL_SCHEMA = pa.DataFrameSchema(
        {
            "timestamp": pa.Column(nullable=False),
            "open": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "high": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "low": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "close": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
            "volume": pa.Column(pl.Float64, checks=pa.Check.ge(0), coerce=True),
            "label": pa.Column(pl.Int32, nullable=False, coerce=True),
            "upper_barrier": pa.Column(pl.Float64, coerce=True),
            "lower_barrier": pa.Column(pl.Float64, coerce=True),
            "touched_bar": pa.Column(pl.Int32, coerce=True),
            "event_end": pa.Column(pl.Int32, coerce=True),
            "sample_weight": pa.Column(pl.Float64, checks=pa.Check.gt(0), coerce=True),
        },
        strict=False,
    )

    @classmethod
    def validate(cls, df: pl.DataFrame, config: object | None = None) -> None:
        """Raise ValueError if label metadata or feature columns are missing."""
        _check_columns(df, cls._STRUCTURAL, cls.__name__)
        cls._STRUCTURAL_SCHEMA.validate(df, lazy=True)
        _validate_monotonic_unique_timestamp(df, cls.__name__)
        if config is not None:
            from thesis.shared.feature_registry import build_feature_output_cols

            feature_cols = set(build_feature_output_cols(config))
            _check_columns(df, feature_cols, f"{cls.__name__}(full)")
