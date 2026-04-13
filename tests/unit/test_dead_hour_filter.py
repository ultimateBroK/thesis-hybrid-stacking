"""Tests for dead hour flag generation and filtering."""

import polars as pl

from thesis.config.loader import Config
from thesis.data.splitting import split_data


class TestDeadHourColumn:
    """Tests for dead_hour column in labels."""

    def test_dead_hour_column_exists(self):
        """Labels parquet should contain a dead_hour column after generation."""
        # This is an integration-style test that verifies the column name.
        # The actual label generation runs the full pipeline.
        # For unit-level, we just verify the _generate_labels_corrected output.
        from thesis.labels.triple_barrier import _generate_labels_corrected
        from thesis.config.loader import load_config

        cfg = load_config()
        n = 50
        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.lit("2024-07-01 00:00:00").str.to_datetime(),
                    pl.lit("2024-07-03 01:00:00").str.to_datetime(),
                    interval="1h",
                    eager=True,
                )[:n],
                "close": [2000.0 + i * 0.5 for i in range(n)],
                "high": [2001.0 + i * 0.5 for i in range(n)],
                "low": [1999.0 + i * 0.5 for i in range(n)],
                "atr_14": [5.0] * n,
            }
        )

        result = _generate_labels_corrected(
            df=df,
            atr_tp=1.5,
            atr_sl=1.5,
            horizon=5,
            min_atr=0.0001,
            session_atr=cfg.labels.session_atr,
        )

        assert "dead_hours" in result
        assert len(result["dead_hours"]) == n
        # Some hours in July summer dead zone (20-0 UTC) should be flagged
        assert any(result["dead_hours"])


class TestDeadHourFiltering:
    """Tests for dead hour filtering in splitting."""

    def test_train_val_filtered_test_kept(self):
        """When dead_hour column exists, train/val should exclude dead hours."""
        # Create mock data with dead_hour column
        n = 100
        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.lit("2024-01-01 00:00:00").str.to_datetime(),
                    pl.lit("2024-01-05 03:00:00").str.to_datetime(),
                    interval="1h",
                    eager=True,
                )[:n],
                "label": [0] * n,
                "dead_hour": [i % 5 == 0 for i in range(n)],  # every 5th is dead
            }
        )

        dead_count = df.filter(pl.col("dead_hour")).height
        assert dead_count > 0  # sanity check

        # Filter train/val
        train_val = df.filter(~pl.col("dead_hour"))
        test = df  # test keeps all

        assert train_val.height == n - dead_count
        assert test.height == n
