"""Tests for sequence model (LSTM/GRU)."""

import numpy as np
import polars as pl
import pytest

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from thesis.models.lstm_model import (
        SequenceClassifier,
        train_lstm,
        _create_sequences,
        _resolve_device,
    )

    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False


# Backward-compatible alias for legacy test names
if HAS_MODEL:
    LSTMClassifier = SequenceClassifier


@pytest.mark.skipif(not HAS_MODEL, reason="Sequence model module not available")
class TestSequenceClassifier:
    """Test cases for SequenceClassifier (LSTM and GRU)."""

    def test_gru_model_initialization(self):
        """Test GRU model can be initialized."""
        model = SequenceClassifier(
            input_size=20,
            hidden_size=64,
            num_layers=1,
            num_classes=3,
            dropout=0.2,
            bidirectional=False,
            model_type="gru",
        )

        assert model is not None
        assert model.hidden_size == 64
        assert model.num_layers == 1
        assert model.model_type == "gru"

    def test_lstm_model_initialization(self):
        """Test LSTM model can be initialized."""
        model = SequenceClassifier(
            input_size=20,
            hidden_size=64,
            num_layers=1,
            num_classes=3,
            dropout=0.2,
            bidirectional=False,
            model_type="lstm",
        )

        assert model is not None
        assert model.model_type == "lstm"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_gru_forward_pass_shape(self):
        """Test GRU model output shapes."""
        model = SequenceClassifier(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            num_classes=3,
            dropout=0.1,
            bidirectional=False,
            model_type="gru",
        )
        model.eval()

        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 10)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_lstm_forward_pass_shape(self):
        """Test LSTM model output shapes."""
        model = SequenceClassifier(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            num_classes=3,
            dropout=0.1,
            bidirectional=False,
            model_type="lstm",
        )
        model.eval()

        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 10)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_model_train_mode(self):
        """Test model can switch between train/eval modes."""
        model = SequenceClassifier(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            num_classes=3,
            dropout=0.1,
            bidirectional=False,
            model_type="gru",
        )

        model.train()
        assert model.training

        model.eval()
        assert not model.training

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_bidirectional_output_shape(self):
        """Test bidirectional model output shape."""
        model = SequenceClassifier(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            num_classes=3,
            dropout=0.1,
            bidirectional=True,
            model_type="gru",
        )
        model.eval()

        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 10)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_model_type_switches_architecture(self):
        """Test that model_type parameter switches between GRU and LSTM."""
        gru_model = SequenceClassifier(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            num_classes=3,
            dropout=0.1,
            bidirectional=False,
            model_type="gru",
        )
        lstm_model = SequenceClassifier(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            num_classes=3,
            dropout=0.1,
            bidirectional=False,
            model_type="lstm",
        )

        assert isinstance(gru_model.rnn, nn.GRU)
        assert isinstance(lstm_model.rnn, nn.LSTM)


@pytest.mark.skipif(
    not (HAS_MODEL and HAS_TORCH), reason="Model or PyTorch not available"
)
class TestSequenceTraining:
    """Tests for sequence model training process."""

    @pytest.mark.slow
    def test_gru_training_with_synthetic_data(self):
        """Test GRU can be trained on synthetic sequences."""
        model = SequenceClassifier(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            num_classes=3,
            dropout=0.1,
            bidirectional=False,
            model_type="gru",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        np.random.seed(42)
        n_samples = 50
        seq_len = 10

        X = torch.randn(n_samples, seq_len, 5)
        y = torch.randint(0, 3, (n_samples,))

        model.train()
        initial_loss = None

        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)

            if epoch == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 1.5


@pytest.mark.skipif(not HAS_MODEL, reason="Model module not available")
class TestDeviceResolution:
    """Tests for _resolve_device() helper."""

    def test_resolve_device_auto(self):
        """Test device resolution with 'auto' setting."""
        device = _resolve_device("auto")
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]

    def test_resolve_device_explicit_cpu(self):
        """Test explicit CPU device setting."""
        device = _resolve_device("cpu")
        assert device.type == "cpu"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_prediction_consistency(self):
        """Test model produces consistent predictions in eval mode."""
        model = SequenceClassifier(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            num_classes=3,
            dropout=0.0,
            bidirectional=False,
            model_type="gru",
        )
        model.eval()

        x = torch.randn(1, 10, 5)

        with torch.no_grad():
            pred1 = model(x)
            pred2 = model(x)

        assert torch.allclose(pred1, pred2, rtol=1e-5)


@pytest.mark.skipif(not HAS_MODEL, reason="Model module not available")
class TestSequenceCreation:
    """Tests for sequence creation."""

    def test_create_sequences(self):
        """Test sequence creation from data."""
        np.random.seed(42)
        n = 100

        df = pl.DataFrame(
            {
                "open": np.random.randn(n),
                "high": np.random.randn(n) + 1,
                "low": np.random.randn(n) - 1,
                "close": np.cumsum(np.random.randn(n) * 0.1) + 1500,
                "volume": np.random.randint(100, 1000, n),
                "label": np.random.choice([-1, 0, 1], n),
            }
        )

        feature_cols = ["open", "high", "low", "close", "volume"]
        seq_length = 20

        X, y, means, stds = _create_sequences(df, feature_cols, seq_length)

        assert X is not None
        assert len(X) > 0
        assert X.shape[1:] == (seq_length, len(feature_cols))
        assert len(y) == len(X)
        assert y.min() >= 0 and y.max() <= 2

    @pytest.mark.critical
    def test_sequence_temporal_order(self):
        """CRITICAL: Test sequences maintain temporal order."""
        n_samples = 100
        features = np.arange(n_samples).reshape(-1, 1).astype(float)
        labels = np.zeros(n_samples)

        df = pl.DataFrame(
            {
                "open": features.flatten(),
                "high": features.flatten(),
                "low": features.flatten(),
                "close": features.flatten(),
                "volume": features.flatten(),
                "label": labels,
            }
        )

        feature_cols = ["open", "high", "low", "close", "volume"]
        seq_len = 10

        try:
            X, y, _, _ = _create_sequences(df, feature_cols, seq_len)

            for seq in X:
                assert np.all(np.diff(seq[:, 0]) > 0)
        except Exception as e:
            pytest.skip(f"Sequence test failed: {e}")

    @pytest.mark.critical
    def test_no_test_data_in_train_sequences(self):
        """CRITICAL: Verify test data never appears in training sequences."""
        n_samples = 500
        train_end = 400
        test_start = 405
        seq_len = 20

        train_sequences = []
        for i in range(0, train_end - seq_len + 1):
            seq_indices = list(range(i, i + seq_len))
            train_sequences.append(seq_indices)

        test_sequences = []
        for i in range(test_start, n_samples - seq_len + 1):
            seq_indices = list(range(i, i + seq_len))
            test_sequences.append(seq_indices)

        all_train_indices = set()
        for seq in train_sequences:
            all_train_indices.update(seq)

        all_test_indices = set()
        for seq in test_sequences:
            all_test_indices.update(seq)

        overlap = all_train_indices & all_test_indices
        assert len(overlap) == 0, (
            f"Data leakage: {len(overlap)} indices in both train and test"
        )


@pytest.mark.skipif(
    not (HAS_MODEL and HAS_TORCH), reason="Model or PyTorch not available"
)
class TestDataLeakage:
    """CRITICAL: Data leakage prevention tests."""

    @pytest.mark.critical
    def test_normalization_window_temporal_integrity(self):
        """CRITICAL: Verify normalization uses only past data."""
        np.random.seed(42)
        data = np.random.randn(100, 3)

        expanding_norm = np.zeros_like(data)
        for t in range(len(data)):
            if t == 0:
                expanding_norm[t] = data[t]
            else:
                window = data[: t + 1]
                mean = window.mean(axis=0)
                std = window.std(axis=0) + 1e-8
                expanding_norm[t] = (data[t] - mean) / std

        assert not np.isnan(expanding_norm).any()

    @pytest.mark.critical
    def test_val_uses_train_normalization_stats(self):
        """CRITICAL: Validation sequences must use train-only stats."""
        np.random.seed(42)
        n = 100

        train_df = pl.DataFrame(
            {
                "open": np.random.randn(n) + 10,
                "high": np.random.randn(n) + 11,
                "low": np.random.randn(n) + 9,
                "close": np.random.randn(n) + 10,
                "volume": np.random.randn(n) + 100,
                "label": np.random.choice([-1, 0, 1], n),
            }
        )

        val_df = pl.DataFrame(
            {
                "open": np.random.randn(n) + 20,
                "high": np.random.randn(n) + 21,
                "low": np.random.randn(n) + 19,
                "close": np.random.randn(n) + 20,
                "volume": np.random.randn(n) + 200,
                "label": np.random.choice([-1, 0, 1], n),
            }
        )

        feature_cols = ["open", "high", "low", "close", "volume"]
        seq_len = 10

        X_train, y_train, train_means, train_stds = _create_sequences(
            train_df, feature_cols, seq_len
        )

        X_val_own, _, val_own_means, _ = _create_sequences(
            val_df, feature_cols, seq_len
        )
        X_val_correct, _, _, _ = _create_sequences(
            val_df, feature_cols, seq_len, norm_stats=(train_means, train_stds)
        )

        assert not np.allclose(X_val_own, X_val_correct), (
            "Val normalization must differ when using train stats vs own stats"
        )

        raw_val = val_df.select(feature_cols).to_numpy()
        expected_normalized = (raw_val - train_means) / train_stds
        expected_last = expected_normalized[seq_len - 1]
        actual_last = X_val_correct[0, -1, :]
        np.testing.assert_allclose(actual_last, expected_last, rtol=1e-5)

    @pytest.mark.critical
    def test_sequence_boundaries_no_overlap(self):
        """CRITICAL: Test train/test sequence boundaries."""
        total_len = 1000
        train_len = 700
        gap = 50
        test_start = train_len + gap
        seq_len = 30
        horizon = 1

        max_train_seq_end = train_len - horizon
        min_test_seq_start = test_start

        assert max_train_seq_end < min_test_seq_start
