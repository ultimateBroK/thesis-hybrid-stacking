"""Tests for configuration consistency between sections.

Verifies that labels and backtest barrier multipliers stay aligned,
since the backtest SL/TP must match the label barriers used to
generate the signals being traded.
"""

from pathlib import Path

import pytest

from thesis._shared.config import Config
from thesis.pipeline import _cache_hash, _resolve_cache_path


# ──────────────────────────────────────────────────────────────────────
# Barrier consistency
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_label_backtest_barrier_consistency() -> None:
    """Default config must have matching label and backtest multipliers."""
    cfg = Config()

    assert cfg.labels.atr_tp_multiplier == cfg.backtest.atr_tp_multiplier, (
        f"labels.atr_tp_multiplier ({cfg.labels.atr_tp_multiplier}) "
        f"!= backtest.atr_tp_multiplier ({cfg.backtest.atr_tp_multiplier})"
    )

    assert cfg.labels.atr_sl_multiplier == cfg.backtest.atr_stop_multiplier, (
        f"labels.atr_sl_multiplier ({cfg.labels.atr_sl_multiplier}) "
        f"!= backtest.atr_stop_multiplier ({cfg.backtest.atr_stop_multiplier})"
    )


# ──────────────────────────────────────────────────────────────────────
# _cache_hash
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCacheHash:
    """Tests for the cache-fingerprinting function ``_cache_hash``."""

    def test_different_configs_different_hashes(self) -> None:
        """Changing a relevant field should change the stage hash."""
        cfg_a = Config()
        cfg_b = Config()
        cfg_b.features.atr_period = 21  # different from default 14

        h_a = _cache_hash(cfg_a, stage_num=2)  # stage 2 depends on "features"
        h_b = _cache_hash(cfg_b, stage_num=2)

        assert h_a, "hash must not be empty"
        assert h_b, "hash must not be empty"
        assert h_a != h_b, (
            f"Hashes should differ for different atr_period:\n  a={h_a}\n  b={h_b}"
        )

    def test_identical_configs_identical_hashes(self) -> None:
        """Two configs with the same values must produce the same hash."""
        cfg_a = Config()
        cfg_b = Config()

        for stage in [1, 2, 3, 4, 5]:
            h_a = _cache_hash(cfg_a, stage)
            h_b = _cache_hash(cfg_b, stage)
            assert h_a == h_b, (
                f"Identical configs must produce identical hashes "
                f"for stage {stage}: {h_a} vs {h_b}"
            )

    def test_hash_ignores_irrelevant_sections(self) -> None:
        """Hash for stage N only depends on sections listed in
        ``_STAGE_CONFIG_SECTIONS[N]``."""
        stage = 2  # depends on ["features"]

        cfg_base = Config()
        h_base = _cache_hash(cfg_base, stage)

        # Change labels — should NOT affect stage 2 hash
        cfg_label_alt = Config()
        cfg_label_alt.labels.atr_tp_multiplier = 999.0
        h_label_alt = _cache_hash(cfg_label_alt, stage)
        assert h_base == h_label_alt, "Stage 2 hash must ignore labels changes"

        # Change model — should NOT affect stage 2 hash
        cfg_model_alt = Config()
        cfg_model_alt.model.learning_rate = 0.999
        h_model_alt = _cache_hash(cfg_model_alt, stage)
        assert h_base == h_model_alt, "Stage 2 hash must ignore model changes"

    def test_empty_sections_returns_empty_string(self) -> None:
        """Stage 6 has no mapped sections, so hash must be empty."""
        cfg = Config()
        h = _cache_hash(cfg, stage_num=6)
        assert h == "", f"Stage 6 hash must be empty, got {h!r}"

    def test_stage_4_depends_on_model_gru_validation(self) -> None:
        """Stage 4 depends on model, gru, and validation sections."""
        cfg_a = Config()
        cfg_b = Config()

        # Change model param
        cfg_b.model.num_leaves = 999
        h_a = _cache_hash(cfg_a, stage_num=4)
        h_b = _cache_hash(cfg_b, stage_num=4)
        assert h_a != h_b, "Stage 4 hash must react to model changes"

        # Change gru param
        cfg_c = Config()
        cfg_c.gru.hidden_size = 256
        h_c = _cache_hash(cfg_c, stage_num=4)
        assert h_a != h_c, "Stage 4 hash must react to gru changes"


# ──────────────────────────────────────────────────────────────────────
# _resolve_cache_path and cache_invalidation
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCacheInvalidation:
    """Tests for ``_resolve_cache_path`` with different invalidation
    strategies: ``"path"``, ``"hash"``, and ``"none"``."""

    BASE = "data/processed/features.parquet"
    STAGE = 2
    CONFIG = Config()

    def test_invalidation_none_returns_none(self) -> None:
        """``cache_invalidation="none"`` disables caching entirely."""
        result = _resolve_cache_path(self.BASE, "none", self.CONFIG, self.STAGE)
        assert result is None, "cache_invalidation='none' must return None"

    def test_invalidation_path_returns_unmodified(self) -> None:
        """``cache_invalidation="path"`` returns the path as-is."""
        result = _resolve_cache_path(self.BASE, "path", self.CONFIG, self.STAGE)
        assert result == Path(self.BASE), (
            f"cache_invalidation='path' must return unmodified path, got {result}"
        )

    def test_invalidation_hash_embeds_fingerprint(self) -> None:
        """``cache_invalidation="hash"`` appends a hash to the filename
        stem for stages with mapped config sections."""
        result = _resolve_cache_path(self.BASE, "hash", self.CONFIG, self.STAGE)
        assert result is not None, "hash invalidation must return a path"
        assert result != self.BASE, "hash invalidation must alter the path"
        # The stem should be "features_XXXXXXXX" where XXXXXXXX is 8-hex
        stem = result.stem
        assert stem.startswith("features_"), (
            f"Expected stem to start with 'features_', got {stem!r}"
        )
        suffix = stem[len("features_") :]
        assert len(suffix) == 8, (
            f"Expected 8-char hex suffix, got {len(suffix)} chars: {suffix!r}"
        )
        # Verify it's hex
        int(suffix, 16)

    def test_invalidation_hash_stage_6_no_append(self) -> None:
        """Stage 6 has no mapped sections, so hash invalidation falls
        back to the unmodified path."""
        result = _resolve_cache_path(self.BASE, "hash", self.CONFIG, stage_num=6)
        assert result == Path(self.BASE), (
            "Stage 6 (no sections) should return unmodified path with hash invalidation"
        )

    def test_base_none_returns_none(self) -> None:
        """When ``base`` is ``None``, return ``None`` regardless of
        invalidation strategy."""
        for strategy in ("path", "hash", "none"):
            result = _resolve_cache_path(None, strategy, self.CONFIG, self.STAGE)
            assert result is None, (
                f"base=None with invalidation={strategy!r} must return None"
            )

    def test_hash_invalidation_reacts_to_config_change(self) -> None:
        """With 'hash' invalidation, a config change should produce a
        different path."""
        cfg_base = Config()
        cfg_mod = Config()
        cfg_mod.features.atr_period = 21

        p_base = _resolve_cache_path(self.BASE, "hash", cfg_base, self.STAGE)
        p_mod = _resolve_cache_path(self.BASE, "hash", cfg_mod, self.STAGE)

        assert p_base != p_mod, (
            f"Hash-invalidated paths must differ for different configs:\n"
            f"  base={p_base}\n  mod={p_mod}"
        )
