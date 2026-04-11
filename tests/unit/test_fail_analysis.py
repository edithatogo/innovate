"""Tests for fail/analysis module."""

import numpy as np
import pytest

from innovate.fail.analysis import analyze_failure


class TestAnalyzeFailure:
    """Test failure analysis function."""

    def test_no_failures(self):
        """Test when no technologies fail."""
        predictions = np.array([
            [0.15, 0.20],
            [0.25, 0.30],
            [0.35, 0.40],
        ])
        failed = analyze_failure(predictions, failure_threshold=0.1)
        assert failed == []

    def test_all_fail(self):
        """Test when all technologies fail."""
        predictions = np.array([
            [0.01, 0.02],
            [0.03, 0.04],
            [0.05, 0.06],
        ])
        failed = analyze_failure(predictions, failure_threshold=0.1)
        assert failed == [0, 1]

    def test_partial_failures(self):
        """Test when some technologies fail."""
        predictions = np.array([
            [0.15, 0.01, 0.05],
            [0.25, 0.02, 0.08],
            [0.35, 0.03, 0.09],
        ])
        failed = analyze_failure(predictions, failure_threshold=0.1)
        assert failed == [1, 2]

    def test_custom_threshold(self):
        """Test with custom failure threshold."""
        predictions = np.array([
            [0.05, 0.25],
            [0.08, 0.35],
        ])
        failed = analyze_failure(predictions, failure_threshold=0.2)
        assert failed == [0]

    def test_custom_time_horizon(self):
        """Test with custom time horizon."""
        predictions = np.array([
            [0.01, 0.02],
            [0.02, 0.03],
            [0.15, 0.20],  # Would pass if full horizon
        ])
        failed = analyze_failure(predictions, failure_threshold=0.1, time_horizon=2)
        assert failed == [0, 1]

    def test_full_time_horizon(self):
        """Test with time_horizon=-1 (full series)."""
        predictions = np.array([
            [0.01, 0.02],
            [0.05, 0.10],
            [0.09, 0.20],
        ])
        failed = analyze_failure(predictions, failure_threshold=0.1, time_horizon=-1)
        assert failed == [0]

    def test_not_2d_raises(self):
        """Test that non-2D array raises ValueError."""
        with pytest.raises(ValueError, match="2D array"):
            analyze_failure(np.array([0.1, 0.2, 0.3]))

    def test_invalid_threshold_raises(self):
        """Test that invalid threshold raises ValueError."""
        predictions = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError, match="between 0 and 1"):
            analyze_failure(predictions, failure_threshold=0.0)
        with pytest.raises(ValueError, match="between 0 and 1"):
            analyze_failure(predictions, failure_threshold=1.5)

    def test_invalid_time_horizon_raises(self):
        """Test that invalid time_horizon raises ValueError."""
        predictions = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError, match="time_horizon"):
            analyze_failure(predictions, time_horizon=5)

    def test_empty_predictions(self):
        """Test with empty 2D array."""
        predictions = np.array([]).reshape(0, 2)
        with pytest.raises(ValueError, match="time_horizon"):
            analyze_failure(predictions)

    def test_single_time_step(self):
        """Test with single time step."""
        predictions = np.array([[0.05, 0.15]])
        failed = analyze_failure(predictions, failure_threshold=0.1)
        assert failed == [0]

    def test_single_technology(self):
        """Test with single technology."""
        predictions = np.array([[0.05], [0.08], [0.12]])
        failed = analyze_failure(predictions, failure_threshold=0.1)
        assert failed == []  # Eventually exceeds threshold
