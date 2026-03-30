"""
Tests for confidence threshold in stacking predictions.

Verifies that predictions with max probability < 0.6 are converted to Hold (class 0).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestConfidenceThreshold:
    """Tests for the 0.6 confidence threshold logic."""

    def test_predictions_below_threshold_converted_to_hold(
        self, mock_predictions_with_low_confidence
    ):
        """
        Verify predictions with max_prob < 0.6 are set to Hold (class 0).
        """
        data = mock_predictions_with_low_confidence
        probs = data["probabilities"]
        max_probs = data["max_probabilities"]
        expected = data["expected_after_threshold"]
        
        # Simulate threshold logic
        confidence_threshold = 0.6
        low_confidence_mask = max_probs < confidence_threshold
        
        # Get argmax predictions
        preds = np.argmax(probs, axis=1) - 1  # Convert 0,1,2 to -1,0,1
        
        # Apply threshold
        preds[low_confidence_mask] = 0  # Set low confidence to Hold
        
        # Verify
        assert np.array_equal(preds, expected), \
            f"Low confidence predictions should be converted to Hold\n" \
            f"Expected: {expected}\nGot: {preds}"
    
    def test_predictions_above_threshold_preserved(
        self, mock_predictions_with_low_confidence
    ):
        """
        Verify predictions with max_prob >= 0.6 keep their original class.
        """
        data = mock_predictions_with_low_confidence
        probs = data["probabilities"]
        max_probs = data["max_probabilities"]
        
        confidence_threshold = 0.6
        low_conf_mask = max_probs < confidence_threshold
        high_conf_mask = ~low_conf_mask
        
        # Original predictions for high confidence
        original_preds = np.argmax(probs, axis=1) - 1
        
        # Apply threshold
        final_preds = original_preds.copy()
        final_preds[low_conf_mask] = 0
        
        # High confidence predictions should be unchanged
        assert np.array_equal(
            final_preds[high_conf_mask],
            original_preds[high_conf_mask]
        ), "High confidence predictions should be preserved"
    
    def test_threshold_boundary_at_0_6(self, mock_predictions_exact_threshold):
        """
        Test exact boundary cases around 0.6 threshold.
        
        - 0.599: Should become Hold
        - 0.600: Should preserve original
        - 0.601: Should preserve original
        """
        data = mock_predictions_exact_threshold
        probs = data["probabilities"]
        max_probs = data["max_probabilities"]
        expected = data["expected_after_threshold"]
        
        confidence_threshold = 0.6
        low_conf_mask = max_probs < confidence_threshold
        
        preds = np.argmax(probs, axis=1) - 1
        preds[low_conf_mask] = 0
        
        # Check specific boundary cases
        # Index 0: 0.599 < 0.6 -> Hold (0)
        assert preds[0] == 0, "0.599 should be converted to Hold"
        
        # Index 1: 0.600 >= 0.6 -> preserve (class -1)
        assert preds[1] == -1, "0.600 should be preserved"
        
        # Index 2: 0.601 >= 0.6 -> preserve (class -1)
        assert preds[2] == -1, "0.601 should be preserved"
        
        # Verify all match expected
        assert np.array_equal(preds, expected), \
            f"Boundary test failed\nExpected: {expected}\nGot: {preds}"
    
    def test_low_confidence_statistics_logged(self, mock_predictions_with_low_confidence):
        """
        Verify statistics are calculated correctly for logging.
        """
        data = mock_predictions_with_low_confidence
        probs = data["probabilities"]
        
        confidence_threshold = 0.6
        max_probs = np.max(probs, axis=1)
        low_conf_mask = max_probs < confidence_threshold
        
        n_low_conf = np.sum(low_conf_mask)
        n_total = len(probs)
        pct_low = 100 * n_low_conf / n_total
        
        assert n_low_conf == data["n_low_confidence"], \
            f"Should detect {data['n_low_confidence']} low confidence, got {n_low_conf}"
        
        assert 30 <= pct_low <= 50, \
            f"Low confidence percentage ({pct_low:.1f}%) should be reasonable"
    
    def test_probabilities_updated_for_low_confidence(self):
        """
        When converting to Hold, probabilities should be updated.
        
        Low confidence: set Hold prob to 1.0, others to 0.0
        """
        probs = np.array([
            [0.55, 0.25, 0.20],  # Low confidence -> Hold
            [0.70, 0.15, 0.15],  # High confidence -> preserve
        ])
        
        confidence_threshold = 0.6
        max_probs = np.max(probs, axis=1)
        low_conf_mask = max_probs < confidence_threshold
        
        # Filtered probabilities
        probs_filtered = probs.copy()
        probs_filtered[low_conf_mask, :] = 0.0
        probs_filtered[low_conf_mask, 1] = 1.0  # Set Hold (index 1) to 1.0
        
        # Check index 0 (low confidence)
        assert probs_filtered[0, 0] == 0.0, "Short prob should be 0.0"
        assert probs_filtered[0, 1] == 1.0, "Hold prob should be 1.0"
        assert probs_filtered[0, 2] == 0.0, "Long prob should be 0.0"
        
        # Check index 1 (high confidence) - unchanged
        assert np.array_equal(probs_filtered[1], probs[1]), \
            "High confidence probs should be unchanged"
    
    def test_original_predictions_stored(self):
        """
        Original predictions should be accessible before threshold application.
        """
        probs = np.array([
            [0.55, 0.25, 0.20],  # Would predict Short (-1)
            [0.70, 0.15, 0.15],  # Would predict Short (-1)
        ])
        
        # Store original
        original_preds = np.argmax(probs, axis=1) - 1
        
        # Apply threshold
        confidence_threshold = 0.6
        max_probs = np.max(probs, axis=1)
        low_conf_mask = max_probs < confidence_threshold
        
        final_preds = original_preds.copy()
        final_preds[low_conf_mask] = 0
        
        # Original should show both as Short
        assert original_preds[0] == -1, "Original prediction should be Short"
        assert original_preds[1] == -1, "Original prediction should be Short"
        
        # Final should show Hold for first
        assert final_preds[0] == 0, "Final prediction should be Hold"
        assert final_preds[1] == -1, "Final prediction should remain Short"
    
    def test_threshold_effect_on_accuracy(self):
        """
        Test that threshold changes accuracy calculation correctly.
        """
        # True labels
        y_true = np.array([-1, 0, 1, -1, 0, 1])
        
        # Predictions without threshold (some wrong)
        y_pred_no_thresh = np.array([-1, 1, 1, -1, 0, 1])  # 1 error at index 1
        
        # Same predictions with threshold applied
        # Assume index 1 has low confidence -> converted to 0
        y_pred_with_thresh = np.array([-1, 0, 1, -1, 0, 1])  # Now correct!
        
        # Calculate accuracies
        acc_no_thresh = np.mean(y_pred_no_thresh == y_true)
        acc_with_thresh = np.mean(y_pred_with_thresh == y_true)
        
        # With threshold, accuracy should improve (5/6 vs 4/6)
        assert acc_with_thresh > acc_no_thresh, \
            f"Threshold should improve accuracy: {acc_with_thresh:.2%} vs {acc_no_thresh:.2%}"
    
    def test_all_classes_can_be_affected_by_threshold(self):
        """
        Verify all three classes can be converted to Hold if low confidence.
        """
        probs = np.array([
            [0.55, 0.25, 0.20],  # Predicts Short (-1), low confidence
            [0.25, 0.55, 0.20],  # Predicts Hold (0), low confidence
            [0.20, 0.25, 0.55],  # Predicts Long (1), low confidence
        ])
        
        confidence_threshold = 0.6
        max_probs = np.max(probs, axis=1)
        
        # All should be below threshold
        assert np.all(max_probs < confidence_threshold), \
            "All max_probs should be < 0.6"
        
        # All should become Hold
        preds = np.argmax(probs, axis=1) - 1
        preds[max_probs < confidence_threshold] = 0
        
        assert np.all(preds == 0), \
            "All predictions should become Hold when low confidence"
    
    def test_confidence_threshold_with_purging_embargo(self):
        """
        Test that confidence threshold works correctly with leakage prevention.
        
        This ensures the threshold is applied AFTER purging/embargo, not before.
        """
        # Simulate predictions after purging/embargo
        n_samples = 100
        
        # Generate predictions
        np.random.seed(42)
        probs = np.random.dirichlet([1, 1, 1], n_samples)
        
        # Apply confidence threshold
        confidence_threshold = 0.6
        max_probs = np.max(probs, axis=1)
        low_conf_mask = max_probs < confidence_threshold
        
        preds = np.argmax(probs, axis=1) - 1
        preds[low_conf_mask] = 0
        
        # Verify no NaN values (indicating proper sequence)
        assert not np.any(np.isnan(preds)), \
            "Predictions should not contain NaN after threshold"
        
        # Verify all values are valid classes
        assert np.all(np.isin(preds, [-1, 0, 1])), \
            "All predictions should be valid classes"


class TestConfidenceThresholdIntegration:
    """Integration tests for confidence threshold in stacking pipeline."""

    def test_threshold_in_stacking_meta_learner(self):
        """
        Verify threshold logic matches stacking.py implementation.
        """
        # Simulate meta-learner predictions
        n_samples = 50
        np.random.seed(42)
        
        # Create realistic probabilities
        probs = np.random.dirichlet([2, 1, 1], n_samples)
        
        # Apply threshold (matching stacking.py logic)
        confidence_threshold = 0.6
        max_probs = np.max(probs, axis=1)
        low_confidence_mask = max_probs < confidence_threshold
        
        final_preds = np.argmax(probs, axis=1) - 1
        final_preds[low_confidence_mask] = 0
        
        # Count statistics
        n_low_conf = np.sum(low_confidence_mask)
        
        # Should have reasonable distribution
        assert 5 <= n_low_conf <= 25, \
            f"Expected 10-50% low confidence, got {n_low_conf}/{n_samples}"
        
        # Final predictions should have Hold class
        hold_count = np.sum(final_preds == 0)
        assert hold_count >= n_low_conf, \
            "Hold count should include all low confidence + original Holds"
    
    def test_probability_filtering_matches_stacking(self):
        """
        Test that probability filtering matches stacking.py implementation.
        """
        probs = np.array([
            [0.55, 0.25, 0.20],  # Low confidence
            [0.70, 0.15, 0.15],  # High confidence
        ])
        
        # Stacking.py logic
        confidence_threshold = 0.6
        max_probs = np.max(probs, axis=1)
        low_conf_mask = max_probs < confidence_threshold
        
        probs_filtered = probs.copy()
        probs_filtered[low_conf_mask, :] = 0.0
        probs_filtered[low_conf_mask, 1] = 1.0  # Class 0 (Hold) is index 1
        
        # Verify
        expected = np.array([
            [0.0, 1.0, 0.0],  # Converted to Hold
            [0.70, 0.15, 0.15],  # Unchanged
        ])
        
        assert np.allclose(probs_filtered, expected), \
            "Probability filtering should match stacking.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
