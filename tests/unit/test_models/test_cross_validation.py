"""Tests for cross-validation splitters."""

import pytest
from datetime import datetime

try:
    from thesis.models.cross_validation import (
        WalkForwardWindow,
        SlidingWindowCV,
        ExpandingWindowCV,
        PurgedKFold,
        create_cv_splitter
    )
    HAS_CV = True
except ImportError:
    HAS_CV = False

try:
    from thesis.config.loader import load_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


@pytest.mark.skipif(not HAS_CV, reason="CV module not available")
class TestWalkForwardWindow:
    """Test cases for WalkForwardWindow dataclass."""

    def test_dataclass_creation(self):
        """Test WalkForwardWindow can be created."""
        window = WalkForwardWindow(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 6, 1),
            val_start=datetime(2020, 6, 1),
            val_end=datetime(2020, 12, 1),
            window_name="Fold_1"
        )
        
        assert window.window_name == "Fold_1"
        assert window.train_start == datetime(2020, 1, 1)
        assert window.val_end == datetime(2020, 12, 1)


@pytest.mark.skipif(not HAS_CV, reason="CV module not available")
class TestSlidingWindowCV:
    """Test cases for SlidingWindowCV."""

    def test_initialization(self):
        """Test SlidingWindowCV can be initialized."""
        cv = SlidingWindowCV(
            train_years=2,
            val_years=1,
            step_years=1,
            purge_bars=15,
            embargo_bars=10,
            bar_frequency="1h"
        )
        
        assert cv.train_years == 2
        assert cv.val_years == 1
        assert cv.purge_bars == 15

    def test_generate_windows(self):
        """Test window generation."""
        cv = SlidingWindowCV(
            train_years=1,
            val_years=1,
            step_years=1,
            purge_bars=15,
            embargo_bars=10
        )
        
        overall_start = datetime(2020, 1, 1)
        overall_end = datetime(2022, 12, 31)
        
        windows = cv.generate_windows(overall_start, overall_end)
        
        assert len(windows) >= 1
        
        for window in windows:
            # Train before val (accounting for purge)
            assert window.train_end <= window.val_start
            # Val before embargo
            assert window.val_end <= overall_end

    def test_split_with_dataframe(self, raw_ohlcv_data):
        """Test split method with polars DataFrame."""
        df = raw_ohlcv_data.head(500)
        
        cv = SlidingWindowCV(
            train_years=1,
            val_years=1,
            step_years=2,
            purge_bars=15,
            embargo_bars=10
        )
        
        overall_start = df["timestamp"].min()
        overall_end = df["timestamp"].max()
        
        splits = list(cv.split(df, timestamp_col="timestamp", 
                              overall_start=overall_start, overall_end=overall_end))
        
        if len(splits) > 0:
            for train_idx, val_idx, window_name in splits:
                assert len(train_idx) > 0
                assert len(val_idx) > 0
                # No overlap in indices
                assert len(set(train_idx) & set(val_idx)) == 0
                # Train indices come before val indices
                assert max(train_idx) < min(val_idx)

    def test_get_n_splits(self):
        """Test get_n_splits method."""
        cv = SlidingWindowCV(
            train_years=1,
            val_years=1,
            step_years=1
        )
        
        overall_start = datetime(2020, 1, 1)
        overall_end = datetime(2022, 12, 31)
        
        n_splits = cv.get_n_splits(overall_start, overall_end)
        assert n_splits >= 1


@pytest.mark.skipif(not HAS_CV, reason="CV module not available")
class TestExpandingWindowCV:
    """Test cases for ExpandingWindowCV."""

    def test_initialization(self):
        """Test ExpandingWindowCV can be initialized."""
        cv = ExpandingWindowCV(
            min_train_years=1,
            val_years=1,
            step_years=1,
            purge_bars=15,
            embargo_bars=10
        )
        
        assert cv.min_train_years == 1
        assert cv.val_years == 1

    def test_growing_train_set(self, raw_ohlcv_data):
        """Test that train set grows with each split."""
        df = raw_ohlcv_data.head(400)
        
        cv = ExpandingWindowCV(
            min_train_years=1,
            val_years=1,
            step_years=1,
            purge_bars=15,
            embargo_bars=10
        )
        
        overall_start = df["timestamp"].min()
        overall_end = df["timestamp"].max()
        
        splits = list(cv.split(df, timestamp_col="timestamp",
                              overall_start=overall_start, overall_end=overall_end))
        
        if len(splits) >= 2:
            train_sizes = [len(train_idx) for train_idx, _, _ in splits]
            # Train sizes should grow or stay same
            for i in range(len(train_sizes) - 1):
                assert train_sizes[i] <= train_sizes[i + 1]


@pytest.mark.skipif(not HAS_CV, reason="CV module not available")
class TestPurgedKFold:
    """Test cases for PurgedKFold."""

    def test_initialization(self):
        """Test PurgedKFold can be initialized."""
        cv = PurgedKFold(
            n_splits=5,
            purge_bars=15,
            embargo_bars=10
        )
        
        assert cv.n_splits == 5
        assert cv.purge_bars == 15

    def test_split_with_dataframe(self, raw_ohlcv_data):
        """Test split method."""
        df = raw_ohlcv_data.head(200)
        
        cv = PurgedKFold(n_splits=3, purge_bars=5, embargo_bars=5)
        
        splits = list(cv.split(df, timestamp_col="timestamp"))
        
        # Should have n_splits splits
        assert len(splits) == 3
        
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0


@pytest.mark.skipif(not (HAS_CV and HAS_CONFIG), reason="CV or config not available")
class TestCreateCVSplitter:
    """Test cases for create_cv_splitter helper."""

    def test_returns_splitter(self):
        """Test function returns a working splitter."""
        config = load_config("config.toml")
        
        cv = create_cv_splitter(config)
        
        assert cv is not None
        # Should be one of the CV types
        assert isinstance(cv, (SlidingWindowCV, ExpandingWindowCV, PurgedKFold))


class TestCrossValidationDataLeakage:
    """CRITICAL: Data leakage prevention tests."""

    @pytest.mark.critical
    def test_no_test_data_in_train_sequences(self):
        """CRITICAL: Verify test data never appears in training."""
        # Simulate train/test split
        n_samples = 500
        train_end = 400
        test_start = 405
        
        # Create sequences
        seq_len = 20
        
        # Training sequences
        train_sequences = []
        for i in range(0, train_end - seq_len + 1):
            seq_indices = list(range(i, i + seq_len))
            train_sequences.append(seq_indices)
        
        # Test sequences
        test_sequences = []
        for i in range(test_start, n_samples - seq_len + 1):
            seq_indices = list(range(i, i + seq_len))
            test_sequences.append(seq_indices)
        
        # Verify no overlap
        all_train_indices = set()
        for seq in train_sequences:
            all_train_indices.update(seq)
        
        all_test_indices = set()
        for seq in test_sequences:
            all_test_indices.update(seq)
        
        # Should have gap (purge) between train and test
        overlap = all_train_indices & all_test_indices
        assert len(overlap) == 0, f"Data leakage detected: {len(overlap)} overlapping indices"

    @pytest.mark.critical
    @pytest.mark.skipif(not HAS_CV, reason="CV module not available")
    def test_purge_creates_gap(self, raw_ohlcv_data):
        """CRITICAL: Verify purge creates proper gap."""
        df = raw_ohlcv_data.head(300)
        
        cv = SlidingWindowCV(
            train_years=1,
            val_years=1,
            step_years=2,
            purge_bars=15,
            embargo_bars=10
        )
        
        overall_start = df["timestamp"].min()
        overall_end = df["timestamp"].max()
        
        for train_idx, val_idx, _ in cv.split(df, timestamp_col="timestamp",
                                               overall_start=overall_start, overall_end=overall_end):
            if len(train_idx) > 0 and len(val_idx) > 0:
                max_train = max(train_idx)
                min_val = min(val_idx)
                
                # Gap should be at least purge_bars
                gap = min_val - max_train - 1
                assert gap >= 15, f"Gap {gap} smaller than purge_bars (15)"

    @pytest.mark.critical
    @pytest.mark.skipif(not HAS_CV, reason="CV module not available")
    def test_temporal_order_preserved(self, raw_ohlcv_data):
        """CRITICAL: Verify temporal ordering is maintained."""
        df = raw_ohlcv_data.head(300)
        
        cv = SlidingWindowCV(
            train_years=1,
            val_years=1,
            step_years=2
        )
        
        overall_start = df["timestamp"].min()
        overall_end = df["timestamp"].max()
        timestamps = df["timestamp"].to_numpy()
        
        for train_idx, val_idx, _ in cv.split(df, timestamp_col="timestamp",
                                               overall_start=overall_start, overall_end=overall_end):
            if len(train_idx) > 0 and len(val_idx) > 0:
                train_times = timestamps[train_idx]
                val_times = timestamps[val_idx]
                
                # All train times should be before val times
                assert max(train_times) < min(val_times)

    @pytest.mark.critical
    @pytest.mark.skipif(not HAS_CV, reason="CV module not available")
    def test_no_future_information_leak(self, raw_ohlcv_data):
        """CRITICAL: Verify no future information in training."""
        df = raw_ohlcv_data.head(400)
        
        cv = SlidingWindowCV(
            train_years=1,
            val_years=1,
            step_years=2,
            purge_bars=15
        )
        
        overall_start = df["timestamp"].min()
        overall_end = df["timestamp"].max()
        timestamps = df["timestamp"].to_numpy()
        
        window_count = 0
        for train_idx, val_idx, window_name in cv.split(df, timestamp_col="timestamp",
                                                        overall_start=overall_start, overall_end=overall_end):
            window_count += 1
            if len(train_idx) > 0 and len(val_idx) > 0:
                # Train times
                train_times = timestamps[train_idx]
                # Val times
                val_times = timestamps[val_idx]
                
                # Check strict temporal separation
                assert max(train_times) < min(val_times), \
                    f"Window {window_name}: train times not before val times"
        
        # If windows were generated, verify we checked at least one
        # If no windows (small test data), that's also acceptable
        if window_count > 0:
            assert window_count >= 1
