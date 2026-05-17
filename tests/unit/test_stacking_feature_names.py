"""Regression tests for stacking prediction feature-name handling."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from thesis.models.stacking import _aligned_proba


@pytest.mark.unit
def test_aligned_predict_proba_preserves_lgbm_feature_names() -> None:
    """LightGBM fitted with DataFrame names should be predicted with names too."""
    import lightgbm as lgb

    feature_names = ["f0", "f1"]
    X = pd.DataFrame(
        {
            "f0": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
            "f1": [1.0, 0.9, 0.0, 0.1, 1.0, 1.1],
        }
    )
    y = np.array([-1, -1, 0, 0, 1, 1], dtype=np.int32)
    model = lgb.LGBMClassifier(
        n_estimators=3,
        min_child_samples=1,
        min_data_in_bin=1,
        verbose=-1,
        random_state=2024,
    ).fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proba = _aligned_proba(model, X.to_numpy(), feature_names)

    assert proba.shape == (len(X), 3)
    assert not any("valid feature names" in str(w.message) for w in caught)
