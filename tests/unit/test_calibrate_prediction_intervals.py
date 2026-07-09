from innovate.advanced_runtime import calibrate_prediction_intervals


def test_calibrate_prediction_intervals_happy_path():
    """Verify calibrate_prediction_intervals computes intervals and coverage correctly."""
    time = [1.0, 2.0, 3.0]
    observed = [10.0, 20.0, 30.0]
    predicted = [9.0, 21.0, 29.0]

    result = calibrate_prediction_intervals(
        time=time,
        observed=observed,
        predicted=predicted,
        confidence=0.5,
    )

    assert result.workflow == "uncertainty_calibration"
    assert result.stability == "stable"
    assert len(result.lower) == 3
    assert len(result.upper) == 3
    assert result.diagnostics["coverage"] > 0
    assert result.metadata["confidence"] == 0.5


import pytest


def test_calibrate_prediction_intervals_invalid_confidence():
    """Verify ValueError is raised when confidence is out of bounds."""
    time = [1.0, 2.0, 3.0]
    observed = [10.0, 20.0, 30.0]
    predicted = [9.0, 21.0, 29.0]

    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        calibrate_prediction_intervals(
            time=time,
            observed=observed,
            predicted=predicted,
            confidence=1.5,
        )

    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        calibrate_prediction_intervals(
            time=time,
            observed=observed,
            predicted=predicted,
            confidence=-0.5,
        )


def test_calibrate_prediction_intervals_mismatched_lengths():
    """Verify ValueError is raised when input array lengths do not match."""
    time = [1.0, 2.0, 3.0]
    observed = [10.0, 20.0, 30.0]

    with pytest.raises(ValueError, match="observed and predicted lengths must match time length"):
        calibrate_prediction_intervals(
            time=time,
            observed=observed,
            predicted=[9.0, 21.0],  # Mismatched length
        )

    with pytest.raises(ValueError, match="observed and predicted lengths must match time length"):
        calibrate_prediction_intervals(
            time=time,
            observed=[10.0, 20.0],  # Mismatched length
            predicted=[9.0, 21.0, 29.0],
        )


def test_calibrate_prediction_intervals_mismatched_holdout_length():
    """Verify ValueError is raised when holdout array length does not match time length."""
    time = [1.0, 2.0, 3.0]
    observed = [10.0, 20.0, 30.0]
    predicted = [9.0, 21.0, 29.0]

    with pytest.raises(ValueError, match="holdout length must match time length"):
        calibrate_prediction_intervals(
            time=time,
            observed=observed,
            predicted=predicted,
            holdout=[1.0, 0.0],  # Mismatched length
        )
