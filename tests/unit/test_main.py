"""Tests for main.py CLI stage resume logic and pipeline guards."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from main import _apply_stage_flags
from thesis.shared.config import Config


class TestStageResumeLogic:
    """Parametrized tests for all 4 stage values (1-4)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("stage", [1, 2, 3, 4])
    def test_stage_disables_correct_flags(self, stage: int) -> None:
        cfg = _apply_stage_flags(Config(), stage)
        flags = [
            cfg.workflow.run_data,
            cfg.workflow.run_dataset,
            cfg.workflow.run_models,
            cfg.workflow.run_reporting,
        ]
        # All flags before `stage` should be False, rest True
        # Stage 1 → all True; Stage N → first N-1 flags False
        for i, flag in enumerate(flags):
            if i < stage - 1:
                assert flag is False, f"Stage {stage}: flag[{i}] should be False"
            else:
                assert flag is True, f"Stage {stage}: flag[{i}] should be True"

    @pytest.mark.unit
    def test_stage_1_enables_all(self) -> None:
        """--stage 1 keeps all workflow flags True."""
        cfg = _apply_stage_flags(Config(), 1)
        assert cfg.workflow.run_data is True
        assert cfg.workflow.run_dataset is True
        assert cfg.workflow.run_models is True
        assert cfg.workflow.run_reporting is True

    @pytest.mark.unit
    def test_stage_4_disables_first_three(self) -> None:
        """--stage 4 disables data, dataset, models but enables reporting."""
        cfg = _apply_stage_flags(Config(), 4)
        assert cfg.workflow.run_data is False
        assert cfg.workflow.run_dataset is False
        assert cfg.workflow.run_models is False
        assert cfg.workflow.run_reporting is True

    @pytest.mark.unit
    def test_stage_flags_reapply_after_session_config_load(self) -> None:
        """--session reload must not reset --stage workflow flags."""
        session_cfg = Config()
        session_cfg.workflow.run_data = True
        session_cfg.workflow.run_dataset = True
        session_cfg.workflow.run_models = True

        result = _apply_stage_flags(session_cfg, 4)

        assert result.workflow.run_data is False
        assert result.workflow.run_dataset is False
        assert result.workflow.run_models is False
        assert result.workflow.run_reporting is True

    @pytest.mark.unit
    def test_force_flag_reapplied_after_session_config_load(self) -> None:
        """--session reloads config, so --force must be applied after reload."""
        from main import _apply_force_flag

        cfg = Config()
        cfg.workflow.force_rerun = False

        result = _apply_force_flag(cfg, force=True)

        assert result.workflow.force_rerun is True


# ---------------------------------------------------------------------------
# Stage numbering contract — --stage N = start at N, continue through 4
# ---------------------------------------------------------------------------


class TestStageNumbering:
    """Tests encoding the --stage CLI contract: --stage N runs stages N..4."""

    @pytest.mark.unit
    def test_stage_1_runs_stages_1_through_4(self) -> None:
        """--stage 1 keeps all four workflow flags True."""
        cfg = _apply_stage_flags(Config(), 1)

        flags = {
            "run_data": cfg.workflow.run_data,
            "run_dataset": cfg.workflow.run_dataset,
            "run_models": cfg.workflow.run_models,
            "run_reporting": cfg.workflow.run_reporting,
        }
        for name, value in flags.items():
            assert value is True, f"--stage 1: {name} must be True"

    @pytest.mark.unit
    def test_stage_3_runs_stages_3_through_4(self) -> None:
        """--stage 3 disables stages 1-2, enables stages 3-4."""
        cfg = _apply_stage_flags(Config(), 3)

        assert cfg.workflow.run_data is False
        assert cfg.workflow.run_dataset is False
        assert cfg.workflow.run_models is True
        assert cfg.workflow.run_reporting is True

    @pytest.mark.unit
    def test_stage_4_runs_only_stage_4(self) -> None:
        """--stage 4 disables stages 1-3, enables only stage 4 reporting."""
        cfg = _apply_stage_flags(Config(), 4)

        assert cfg.workflow.run_reporting is True
        for field in (
            "run_data",
            "run_dataset",
            "run_models",
        ):
            assert getattr(cfg.workflow, field) is False, (
                f"--stage 4: {field} must be False"
            )
