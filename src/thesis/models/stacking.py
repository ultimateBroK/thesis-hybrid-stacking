"""Hybrid Stacking meta-learner."""

import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet

from thesis.config.loader import Config

logger = logging.getLogger("thesis.models")


def train_stacking(config: Config) -> None:
    """Train hybrid stacking meta-learner.

    Combines LightGBM and LSTM predictions using a meta-learner.

    Args:
        config: Configuration object.
    """
    logger.info("Loading base model predictions...")

    # Load predictions
    lgbm_path = Path(config.models["tree"].predictions_path)
    lstm_path = Path(config.models["lstm"].predictions_path)

    if not lgbm_path.exists() or not lstm_path.exists():
        raise FileNotFoundError("Base model predictions not found.")

    lgbm_preds = pl.read_parquet(lgbm_path)
    lstm_preds = pl.read_parquet(lstm_path)

    # Align predictions by timestamp
    logger.info(f"LightGBM predictions: {len(lgbm_preds)} samples")
    logger.info(f"LSTM predictions: {len(lstm_preds)} samples")

    # Join on timestamp to ensure alignment
    aligned = lgbm_preds.join(lstm_preds, on="timestamp", suffix="_lstm")

    logger.info(f"Aligned predictions: {len(aligned)} samples")

    if len(aligned) == 0:
        raise ValueError(
            "No matching timestamps found between LightGBM and LSTM predictions."
        )

    # Create meta-features (6 features: 3 classes × 2 models)
    logger.info("Creating meta-features...")

    # LightGBM probabilities
    lgbm_proba = aligned.select(
        [
            "pred_proba_class_minus1",
            "pred_proba_class_0",
            "pred_proba_class_1",
        ]
    ).to_numpy()

    # LSTM probabilities (with suffix from join)
    lstm_proba = aligned.select(
        [
            "pred_proba_class_minus1_lstm",
            "pred_proba_class_0_lstm",
            "pred_proba_class_1_lstm",
        ]
    ).to_numpy()

    # Concatenate
    X_meta = np.hstack([lgbm_proba, lstm_proba])

    # Labels (use LightGBM's true_label since they should be the same)
    y = aligned["true_label"].to_numpy()

    logger.info(f"Meta-feature shape: {X_meta.shape}")

    # Train meta-learner
    meta_learner_type = config.models["stacking"].meta_learner
    logger.info(f"Training meta-learner: {meta_learner_type}")

    if meta_learner_type == "logistic_regression":
        model = LogisticRegression(
            C=config.models["stacking"].C,
            max_iter=1000,
            random_state=config.workflow.random_seed,
        )
    elif meta_learner_type == "ridge":
        model = Ridge(alpha=config.models["stacking"].alpha)
    elif meta_learner_type == "lasso":
        model = Lasso(alpha=config.models["stacking"].alpha)
    elif meta_learner_type == "elastic_net":
        model = ElasticNet(
            alpha=config.models["stacking"].alpha,
            l1_ratio=config.models["stacking"].l1_ratio,
        )
    elif meta_learner_type == "lightgbm":
        try:
            import lightgbm as lgb

            cfg = config.models["stacking"]
            model = lgb.LGBMClassifier(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                objective="multiclass",
                num_class=3,
                random_state=config.workflow.random_seed,
            )
        except ImportError:
            raise ImportError("LightGBM not installed")
    else:
        raise ValueError(f"Unknown meta-learner: {meta_learner_type}")

    model.fit(X_meta, y)

    # Probability calibration (if enabled)
    if config.models["stacking"].calibrate_probabilities:
        logger.info(
            f"Calibrating probabilities using {config.models['stacking'].calibration_method}"
        )

        if config.models["stacking"].calibration_method == "isotonic":
            calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
        else:  # Platt scaling
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=5)

        calibrated.fit(X_meta, y)
        model = calibrated

    # Save meta-learner
    model_path = Path(config.models["stacking"].model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saved meta-learner: {model_path}")

    # Generate predictions with confidence threshold
    logger.info("Generating stacking predictions...")

    # Confidence threshold: only predict when max probability >= 0.6
    confidence_threshold = 0.6

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_meta)
    else:
        # For regression-based meta-learners, convert to probabilities
        # This is a simplified approach
        preds = model.predict(X_meta)
        probs = np.zeros((len(preds), 3))
        for i, p in enumerate(preds):
            idx = int(p) + 1  # Map -1,0,1 to 0,1,2
            idx = max(0, min(2, idx))
            probs[i, idx] = 1.0

    # Apply confidence threshold: low confidence -> Hold (class 0)
    max_probs = np.max(probs, axis=1)
    low_confidence_mask = max_probs < confidence_threshold

    # Log confidence statistics
    n_low_conf = np.sum(low_confidence_mask)
    logger.info(
        f"Confidence threshold ({confidence_threshold}): {n_low_conf}/{len(probs)} "
        f"({100 * n_low_conf / len(probs):.1f}%) predictions set to Hold (low confidence)"
    )

    # Create final predictions with confidence filtering
    final_preds = np.argmax(probs, axis=1) - 1  # Convert 0,1,2 to -1,0,1
    final_preds[low_confidence_mask] = 0  # Set low confidence to Hold (0)

    # Update probabilities for low confidence cases: set Hold=1.0, others=0.0
    probs_filtered = probs.copy()
    probs_filtered[low_confidence_mask, :] = 0.0
    probs_filtered[low_confidence_mask, 1] = 1.0  # Class 0 (Hold) is index 1

    # Save predictions
    preds_df = pl.DataFrame(
        {
            "pred_proba_class_minus1": probs[:, 0],
            "pred_proba_class_0": probs[:, 1],
            "pred_proba_class_1": probs[:, 2],
        }
    )

    preds_path = Path(config.models["stacking"].meta_predictions_path)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(preds_path)
    logger.info(f"Saved stacking predictions: {preds_path}")


def generate_test_predictions(config: Config) -> None:
    """Generate test set predictions for backtesting.

    Loads trained LightGBM, LSTM, and meta-learner models,
    generates predictions on test data, and saves final stacking predictions.

    Args:
        config: Configuration object.
    """
    import pandas as pd
    import torch
    from torch import nn

    logger.info("Generating test set predictions for backtest...")

    # Load test data
    test_path = Path(config.paths.test_data)
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    test_df = pl.read_parquet(test_path)
    logger.info(f"Loaded test data: {len(test_df)} samples")

    # Get feature columns (same logic as in training)
    exclude_cols = [
        "timestamp",
        "label",
        "tp_price",
        "sl_price",
        "touched_bar",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
    ]
    feature_cols = [c for c in test_df.columns if c not in exclude_cols]

    # ============ LightGBM Predictions ============
    logger.info("Generating LightGBM predictions on test set...")
    lgbm_path = Path(config.models["tree"].model_path)
    if not lgbm_path.exists():
        raise FileNotFoundError(f"LightGBM model not found: {lgbm_path}")

    lgbm_model = joblib.load(lgbm_path)

    # Prepare test features
    X_test = test_df.select(feature_cols).to_numpy()
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)

    # Generate predictions
    lgbm_probs = lgbm_model.predict_proba(X_test_df)
    if hasattr(lgbm_probs, "toarray"):
        lgbm_probs = lgbm_probs.toarray()

    lgbm_preds_df = pl.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "pred_proba_class_minus1": lgbm_probs[:, 0],  # Short
            "pred_proba_class_0": lgbm_probs[:, 1],  # Hold
            "pred_proba_class_1": lgbm_probs[:, 2],  # Long
        }
    )
    logger.info(f"LightGBM test predictions: {len(lgbm_preds_df)} samples")

    # ============ LSTM Predictions ============
    logger.info("Generating LSTM predictions on test set...")
    lstm_path = Path(config.models["lstm"].model_path)
    if not lstm_path.exists():
        raise FileNotFoundError(f"LSTM model not found: {lstm_path}")

    # LSTM uses only OHLCV features - same as training
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    seq_length = config.models["lstm"].sequence_length
    features = test_df.select(ohlcv_cols).to_numpy()

    # Load training normalization statistics (use training stats to prevent data leakage)
    stats_path = Path(config.models["lstm"].model_path).parent / "lstm_norm_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"LSTM normalization stats not found: {stats_path}. Run training first."
        )

    stats = np.load(stats_path)
    train_means = stats["means"]
    train_stds = stats["stds"]

    # Normalize using TRAINING statistics (not test data statistics)
    features = (features - train_means) / train_stds
    logger.info(
        f"Loaded LSTM normalization stats from training: means={train_means.round(2)}, stds={train_stds.round(2)}"
    )

    # Create sequences
    X_test_lstm = []
    for i in range(len(features) - seq_length):
        X_test_lstm.append(features[i : i + seq_length])
    X_test_lstm = np.array(X_test_lstm)

    # Load LSTM model
    device = torch.device(config.models["lstm"].device)

    # LSTM model architecture (must match training)
    class LSTMModel(nn.Module):
        def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            dropout,
            bidirectional,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            lstm_output_dim = hidden_size * (2 if bidirectional else 1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(lstm_output_dim, num_classes)

        def forward(self, x):
            num_directions = 2 if self.bidirectional else 1
            h0 = torch.zeros(
                self.num_layers * num_directions, x.size(0), self.hidden_size
            ).to(x.device)
            c0 = torch.zeros(
                self.num_layers * num_directions, x.size(0), self.hidden_size
            ).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out

    # Model parameters (must match training)
    input_size = len(ohlcv_cols)  # 5 features
    hidden_size = config.models["lstm"].hidden_size
    num_layers = config.models["lstm"].num_layers
    dropout = config.models["lstm"].dropout
    bidirectional = config.models["lstm"].bidirectional
    num_classes = 3

    lstm_model = LSTMModel(
        input_size, hidden_size, num_layers, num_classes, dropout, bidirectional
    )
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.to(device)
    lstm_model.eval()

    # Generate predictions
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_lstm).to(device)
        outputs = lstm_model(X_test_tensor)
        lstm_probs = torch.softmax(outputs, dim=1).cpu().numpy()

    # Get timestamps aligned with sequences (seq_length offset)
    timestamps_lstm = test_df["timestamp"].to_numpy()[seq_length:]

    lstm_preds_df = pl.DataFrame(
        {
            "timestamp": timestamps_lstm,
            "pred_proba_class_minus1_lstm": lstm_probs[:, 0],  # Short
            "pred_proba_class_0_lstm": lstm_probs[:, 1],  # Hold
            "pred_proba_class_1_lstm": lstm_probs[:, 2],  # Long
        }
    )

    # Ensure timestamp has same dtype as LightGBM predictions
    if str(test_df["timestamp"].dtype) != str(lstm_preds_df["timestamp"].dtype):
        lstm_preds_df = lstm_preds_df.with_columns(
            pl.col("timestamp").cast(test_df["timestamp"].dtype)
        )

    logger.info(f"LSTM test predictions: {len(lstm_preds_df)} samples")

    logger.info("Generating stacking predictions on test set...")
    meta_path = Path(config.models["stacking"].model_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta-learner not found: {meta_path}")

    meta_learner = joblib.load(meta_path)

    # Align predictions by timestamp
    aligned = lgbm_preds_df.join(lstm_preds_df, on="timestamp", suffix="_lstm")
    logger.info(f"Aligned test predictions: {len(aligned)} samples")
    if len(aligned) != len(test_df):
        logger.info(
            "Skipped %s warm-up bars before stacking because LSTM uses %s-bar sequences",
            len(test_df) - len(aligned),
            seq_length,
        )

    if len(aligned) == 0:
        raise ValueError(
            "No matching timestamps found between LightGBM and LSTM test predictions."
        )

    # Create meta-features (same as in training)
    X_meta = np.hstack(
        [
            aligned[
                ["pred_proba_class_minus1", "pred_proba_class_0", "pred_proba_class_1"]
            ].to_numpy(),
            aligned[
                [
                    "pred_proba_class_minus1_lstm",
                    "pred_proba_class_0_lstm",
                    "pred_proba_class_1_lstm",
                ]
            ].to_numpy(),
        ]
    )

    # Generate stacking predictions
    if hasattr(meta_learner, "predict_proba"):
        final_probs = meta_learner.predict_proba(X_meta)
    else:
        preds = meta_learner.predict(X_meta)
        final_probs = np.zeros((len(preds), 3))
        for i, p in enumerate(preds):
            idx = int(p) + 1
            idx = max(0, min(2, idx))
            final_probs[i, idx] = 1.0

    # Apply confidence threshold: 0.6 minimum confidence
    confidence_threshold = 0.6
    max_probs = np.max(final_probs, axis=1)
    low_confidence_mask = max_probs < confidence_threshold

    # Log confidence statistics
    n_low_conf = np.sum(low_confidence_mask)
    logger.info(
        f"Confidence threshold ({confidence_threshold}): {n_low_conf}/{len(final_probs)} "
        f"({100 * n_low_conf / len(final_probs):.1f}%) predictions set to Hold (low confidence)"
    )

    # Set low confidence predictions to Hold (class 0)
    final_preds = np.argmax(final_probs, axis=1)
    final_preds[low_confidence_mask] = 1  # Index 1 = Hold (class 0)

    # Update probabilities for low confidence cases
    final_probs_filtered = final_probs.copy()
    final_probs_filtered[low_confidence_mask, :] = 0.0
    final_probs_filtered[low_confidence_mask, 1] = 1.0  # Set Hold prob to 1.0

    # Save final predictions
    final_preds_df = pl.DataFrame(
        {
            "timestamp": aligned["timestamp"],
            "pred_proba_class_minus1": final_probs_filtered[:, 0],  # Short
            "pred_proba_class_0": final_probs_filtered[:, 1],  # Hold
            "pred_proba_class_1": final_probs_filtered[:, 2],  # Long
        }
    )

    final_path = Path(config.paths.final_predictions)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_preds_df.write_parquet(final_path)
    logger.info(f"Saved final test predictions: {final_path}")
    logger.info(f"Total test predictions: {len(final_preds_df)} samples")


def apply_confidence_threshold(
    predictions: np.ndarray, probabilities: np.ndarray, threshold: float = 0.6
) -> tuple[np.ndarray, np.ndarray]:
    """Apply confidence threshold to predictions.

    Predictions with max probability below threshold are converted to Hold (0).

    Args:
        predictions: Array of predicted labels (-1, 0, 1)
        probabilities: Array of probability distributions (n_samples, n_classes)
        threshold: Minimum confidence threshold (default 0.6)

    Returns:
        Tuple of (filtered_predictions, filtered_probabilities)
    """
    # Create copies to avoid modifying inputs
    filtered_preds = predictions.copy()
    filtered_probs = probabilities.copy()

    # Calculate max probability for each prediction
    max_probs = np.max(probabilities, axis=1)

    # Find low confidence predictions
    low_conf_mask = max_probs < threshold

    # Convert low confidence predictions to Hold (class 0)
    filtered_preds[low_conf_mask] = 0

    # Update probabilities: set Hold probability to 1.0 for low confidence
    filtered_probs[low_conf_mask, :] = 0.0
    # Assuming Hold is at index 1 (classes: -1=index 0, 0=index 1, 1=index 2)
    if filtered_probs.shape[1] >= 2:
        filtered_probs[low_conf_mask, 1] = 1.0

    return filtered_preds, filtered_probs
