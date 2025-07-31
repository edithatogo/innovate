# tests/test_failure_analysis.py

import pytest
import numpy as np
from innovate.fail.analysis import analyze_failure


def test_analyze_failure_no_failures():
    """Test analysis with no failed technologies."""
    predictions = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6],
            [0.3, 0.6, 0.9],
        ]
    )
    failed = analyze_failure(predictions, failure_threshold=0.2)
    assert failed == []


def test_analyze_failure_one_failure():
    """Test analysis with one failed technology."""
    predictions = np.array(
        [
            [0.01, 0.2, 0.3],
            [0.02, 0.4, 0.6],
            [0.03, 0.6, 0.9],
        ]
    )
    failed = analyze_failure(predictions, failure_threshold=0.1)
    assert failed == [0]


def test_analyze_failure_multiple_failures():
    """Test analysis with multiple failed technologies."""
    predictions = np.array(
        [
            [0.01, 0.05, 0.3],
            [0.02, 0.06, 0.6],
            [0.03, 0.07, 0.9],
        ]
    )
    failed = analyze_failure(predictions, failure_threshold=0.1)
    assert failed == [0, 1]


def test_analyze_failure_with_time_horizon():
    """Test analysis with a specific time horizon."""
    predictions = np.array(
        [
            [0.01, 0.2, 0.3],
            [0.02, 0.4, 0.6],
            [0.15, 0.6, 0.9],  # Tech 0 crosses threshold at t=2
        ]
    )
    # Fails within the first 2 time steps
    failed = analyze_failure(predictions, failure_threshold=0.1, time_horizon=2)
    assert failed == [0]

    # Succeeds over the full time horizon
    failed_full = analyze_failure(predictions, failure_threshold=0.1)
    assert failed_full == []


def test_analyze_failure_invalid_predictions():
    """Test with invalid predictions array."""
    with pytest.raises(ValueError):
        analyze_failure(np.array([0.1, 0.2, 0.3]))


def test_analyze_failure_invalid_threshold():
    """Test with an invalid failure threshold."""
    predictions = np.random.rand(10, 2)
    with pytest.raises(ValueError):
        analyze_failure(predictions, failure_threshold=1.5)
    with pytest.raises(ValueError):
        analyze_failure(predictions, failure_threshold=-0.5)


def test_analyze_failure_invalid_time_horizon():
    """Test with an invalid time horizon."""
    predictions = np.random.rand(10, 2)
    with pytest.raises(ValueError):
        analyze_failure(predictions, time_horizon=0)
    with pytest.raises(ValueError):
        analyze_failure(predictions, time_horizon=20)
