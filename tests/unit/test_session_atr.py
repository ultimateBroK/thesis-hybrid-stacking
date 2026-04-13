"""Tests for DST detection and session-aware ATR multipliers."""

from datetime import datetime, timezone

from thesis.config.loader import (
    SessionATRConfig,
    SessionDefinition,
    load_config,
)
from thesis.labels.triple_barrier import _get_session_atr_multiplier, _is_dst_ny


# ---------------------------------------------------------------------------
# DST detection
# ---------------------------------------------------------------------------


class TestIsDstNy:
    """Tests for _is_dst_ny()."""

    def test_july_is_dst(self):
        """July is summer time (DST active)."""
        assert _is_dst_ny(datetime(2024, 7, 15, 12, 0)) is True

    def test_january_not_dst(self):
        """January is standard time."""
        assert _is_dst_ny(datetime(2024, 1, 15, 12, 0)) is False

    def test_march_before_transition(self):
        """Early March before DST switch is standard time."""
        assert _is_dst_ny(datetime(2024, 3, 5, 12, 0)) is False

    def test_march_after_transition(self):
        """Late March after DST switch is summer time."""
        assert _is_dst_ny(datetime(2024, 3, 20, 12, 0)) is True

    def test_november_before_transition(self):
        """Early November before DST ends is summer time."""
        assert _is_dst_ny(datetime(2024, 11, 1, 12, 0)) is True

    def test_november_after_transition(self):
        """Late November after DST ends is standard time."""
        assert _is_dst_ny(datetime(2024, 11, 15, 12, 0)) is False

    def test_dst_boundary_2024(self):
        """DST 2024 starts March 10, ends Nov 3."""
        assert _is_dst_ny(datetime(2024, 3, 10, 8, 0)) is True  # after switch
        assert _is_dst_ny(datetime(2024, 3, 10, 5, 0)) is False  # before switch

    def test_leap_year(self):
        """DST works correctly in leap years."""
        assert _is_dst_ny(datetime(2020, 2, 29, 12, 0)) is False
        assert _is_dst_ny(datetime(2020, 7, 15, 12, 0)) is True

    def test_timezone_aware_utc(self):
        """DST handles timezone-aware (UTC) timestamps without TypeError."""
        assert _is_dst_ny(datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)) is True
        assert _is_dst_ny(datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)) is False

    def test_timezone_aware_matches_naive(self):
        """Aware and naive timestamps produce the same DST result."""
        naive = datetime(2024, 7, 15, 12, 0)
        aware = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)
        assert _is_dst_ny(naive) == _is_dst_ny(aware)


# ---------------------------------------------------------------------------
# Session ATR multiplier lookup
# ---------------------------------------------------------------------------


def _make_config(
    summer: list[dict] | None = None,
    winter: list[dict] | None = None,
) -> SessionATRConfig:
    """Build a SessionATRConfig with simplified defaults."""
    if summer is None:
        summer = [
            {
                "session": "dead",
                "start_utc": 20,
                "end_utc": 0,
                "tp_mult": 0.0,
                "sl_mult": 0.0,
            },
            {
                "session": "active",
                "start_utc": 0,
                "end_utc": 20,
                "tp_mult": 2.0,
                "sl_mult": 2.0,
            },
        ]
    if winter is None:
        winter = [
            {
                "session": "dead",
                "start_utc": 21,
                "end_utc": 0,
                "tp_mult": 0.0,
                "sl_mult": 0.0,
            },
            {
                "session": "active",
                "start_utc": 0,
                "end_utc": 21,
                "tp_mult": 1.5,
                "sl_mult": 1.5,
            },
        ]
    return SessionATRConfig(
        enabled=True,
        summer=[SessionDefinition(**s) for s in summer],
        winter=[SessionDefinition(**w) for w in winter],
    )


class TestGetSessionATRMultiplier:
    """Tests for _get_session_atr_multiplier()."""

    def test_summer_dead_hour(self):
        cfg = _make_config()
        tp, sl, dead = _get_session_atr_multiplier(22, cfg, is_dst=True)
        assert dead is True
        assert tp == 0.0
        assert sl == 0.0

    def test_summer_active_hour(self):
        cfg = _make_config()
        tp, sl, dead = _get_session_atr_multiplier(10, cfg, is_dst=True)
        assert dead is False
        assert tp == 2.0

    def test_winter_dead_hour(self):
        cfg = _make_config()
        tp, sl, dead = _get_session_atr_multiplier(22, cfg, is_dst=False)
        assert dead is True

    def test_winter_active_hour(self):
        cfg = _make_config()
        tp, sl, dead = _get_session_atr_multiplier(10, cfg, is_dst=False)
        assert dead is False
        assert tp == 1.5

    def test_midnight_wrap_summer(self):
        """UTC 0 should be 'active' in summer (wrap-around from dead 20-0)."""
        cfg = _make_config()
        tp, sl, dead = _get_session_atr_multiplier(0, cfg, is_dst=True)
        assert dead is False
        assert tp == 2.0

    def test_midnight_wrap_winter(self):
        """UTC 0 should be 'active' in winter (wrap-around from dead 21-0)."""
        cfg = _make_config()
        tp, sl, dead = _get_session_atr_multiplier(0, cfg, is_dst=False)
        assert dead is False
        assert tp == 1.5


class TestSessionATRConfigLoad:
    """Tests for loading session ATR config from TOML."""

    def test_config_loads_sessions(self):
        cfg = load_config()
        assert cfg.labels.session_atr.enabled is True
        assert len(cfg.labels.session_atr.summer) == 6
        assert len(cfg.labels.session_atr.winter) == 6

    def test_config_summer_overlap(self):
        cfg = load_config()
        overlap = [s for s in cfg.labels.session_atr.summer if s.session == "overlap"]
        assert len(overlap) == 1
        assert overlap[0].tp_mult == 2.0
        assert overlap[0].start_utc == 12
        assert overlap[0].end_utc == 16

    def test_config_winter_dead(self):
        cfg = load_config()
        dead = [s for s in cfg.labels.session_atr.winter if s.session == "dead"]
        assert len(dead) == 1
        assert dead[0].tp_mult == 0.0
        assert dead[0].start_utc == 21
