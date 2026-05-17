"""Config contract tests."""

from pathlib import Path

import pytest

from thesis.pipeline import _cache_hash, _resolve_cache_path
from thesis.shared.config import Config, load_config
from thesis.shared.constants import CORE_STATIC_FEATURES


@pytest.mark.unit
def test_label_backtest_barriers_match() -> None:
    """Signals and trades must use the same ATR barriers."""
    cfg = Config()

    assert cfg.labels.atr_tp_multiplier == cfg.backtest.atr_tp_multiplier
    assert cfg.labels.atr_sl_multiplier == cfg.backtest.atr_stop_multiplier


@pytest.mark.unit
def test_minimal_public_config_uses_hidden_defaults(tmp_path: Path) -> None:
    """A short TOML file should still produce a complete config."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[data]
symbol = "XAUUSD"
timeframe = "1H"

[model]
architecture = "stacking"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.data_range.start == "2021-01-01"
    assert cfg.data_range.end == "2026-04-30"
    assert cfg.features.static_feature_cols == list(CORE_STATIC_FEATURES)
    assert cfg.model.architecture == "stacking"
    assert cfg.model.stacking_meta_model == "logistic_regression"
    assert cfg.paths.model == "models/lightgbm_model.pkl"


@pytest.mark.unit
def test_unknown_config_keys_fail_fast(tmp_path: Path) -> None:
    """Typos in public config should not be ignored."""
    config_path = tmp_path / "bad_config.toml"
    config_path.write_text(
        """
[data]
symbol = "XAUUSD"
timeframe_typo = "1H"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Unknown config key\(s\) in \[data\]"):
        load_config(config_path)


@pytest.mark.unit
def test_legacy_top_level_multi_timeframe_is_migrated_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Old session snapshots used top-level [multi_timeframe]; keep them readable."""
    config_path = tmp_path / "legacy_config.toml"
    config_path.write_text(
        """
[features]
rsi_period = 14

[multi_timeframe]
atr_short_period = 7
return_lookbacks = [1, 2]
""".strip(),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger="thesis.config"):
        cfg = load_config(config_path)

    assert cfg.features.multi_timeframe.atr_short_period == 7
    assert cfg.features.multi_timeframe.return_lookbacks == [1, 2]
    assert "Ignoring unknown config section" not in caplog.text


@pytest.mark.unit
class TestCacheHash:
    """Stage cache fingerprints."""

    def test_relevant_change_changes_hash(self) -> None:
        """Relevant edits should change a stage hash."""
        cfg_a = Config()
        cfg_b = Config()
        cfg_b.features.atr_period = 21

        h_a = _cache_hash(cfg_a, stage_num=2)
        h_b = _cache_hash(cfg_b, stage_num=2)

        assert h_a
        assert h_b
        assert h_a != h_b

    def test_identical_configs_match(self) -> None:
        """Same values should produce same hashes."""
        cfg_a = Config()
        cfg_b = Config()

        for stage in [1, 2, 3, 4]:
            assert _cache_hash(cfg_a, stage) == _cache_hash(cfg_b, stage)

    def test_stage_hash_ignores_unmapped_sections(self) -> None:
        """A stage hash should ignore unrelated sections."""
        stage = 2
        cfg = Config()
        h_base = _cache_hash(cfg, stage)

        # Stage 2 maps features + labels, so model changes are ignored
        cfg_model_alt = Config()
        cfg_model_alt.model.learning_rate = 0.999
        assert h_base == _cache_hash(cfg_model_alt, stage)

    def test_unmapped_stage_has_empty_hash(self) -> None:
        """Stages without mapped sections should not hash."""
        assert _cache_hash(Config(), stage_num=5) == ""

    def test_stage_3_tracks_training_sections(self) -> None:
        """Training hash should track tabular model and stacking edits."""
        cfg_a = Config()

        cfg_b = Config()
        cfg_b.model.num_leaves = 999
        assert _cache_hash(cfg_a, 3) != _cache_hash(cfg_b, 3)

        cfg_c = Config()
        cfg_c.model.stacking_meta_fraction = 0.25
        assert _cache_hash(cfg_a, 3) != _cache_hash(cfg_c, 3)


@pytest.mark.unit
class TestCacheInvalidation:
    """Cache path strategies."""

    base = "data/processed/features.parquet"
    stage = 2
    cfg = Config()

    def test_none_disables_cache(self) -> None:
        """Strategy none should skip cache reads."""
        assert _resolve_cache_path(self.base, "none", self.cfg, self.stage) is None

    def test_path_keeps_base_path(self) -> None:
        """Strategy path should reuse the base path."""
        assert _resolve_cache_path(self.base, "path", self.cfg, self.stage) == Path(
            self.base
        )

    def test_hash_adds_fingerprint(self) -> None:
        """Strategy hash should append an 8-char hex suffix."""
        result = _resolve_cache_path(self.base, "hash", self.cfg, self.stage)

        assert result is not None
        assert result != self.base
        assert result.stem.startswith("features_")
        assert len(result.stem.removeprefix("features_")) == 8
        int(result.stem.removeprefix("features_"), 16)

    def test_hash_without_stage_sections_keeps_base_path(self) -> None:
        """Strategy hash should no-op for unmapped stages."""
        assert _resolve_cache_path(self.base, "hash", self.cfg, 6) == Path(self.base)

    def test_missing_base_returns_none(self) -> None:
        """A missing base path should stay missing."""
        for strategy in ("path", "hash", "none"):
            assert _resolve_cache_path(None, strategy, self.cfg, self.stage) is None

    def test_hash_path_changes_with_config(self) -> None:
        """Hashed paths should change when config changes."""
        cfg_base = Config()
        cfg_mod = Config()
        cfg_mod.features.atr_period = 21

        p_base = _resolve_cache_path(self.base, "hash", cfg_base, self.stage)
        p_mod = _resolve_cache_path(self.base, "hash", cfg_mod, self.stage)

        assert p_base != p_mod
