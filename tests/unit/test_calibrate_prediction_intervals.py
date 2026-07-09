import pytest

from innovate.advanced_runtime import calibrate_prediction_intervals


def test_calibrate_prediction_intervals_happy_path():
    """Verify basic interval calibration logic."""
    time = [1.0, 2.0, 3.0, 4.0, 5.0]
    observed = [10.0, 12.0, 15.0, 14.0, 18.0]
    predicted = [10.5, 11.5, 14.0, 15.0, 17.0]

    # Residuals: actual - forecast
    # [10.0-10.5, 12.0-11.5, 15.0-14.0, 14.0-15.0, 18.0-17.0]
    # [-0.5, 0.5, 1.0, -1.0, 1.0]
    # Absolute residuals sorted: [0.5, 0.5, 1.0, 1.0, 1.0]
    # Length of residuals is 5. Max index is 4.
    # Confidence 0.8: round(0.8 * 4) = round(3.2) = 3
    # absolute_residuals[3] = 1.0
    # half_width = 1.0

    result = calibrate_prediction_intervals(time=time, observed=observed, predicted=predicted, confidence=0.8)

    assert result.workflow == "uncertainty_calibration"
    assert result.lower == [9.5, 10.5, 13.0, 14.0, 16.0]
    assert result.upper == [11.5, 12.5, 15.0, 16.0, 18.0]
    # Coverage:
    # 9.5 <= 10.0 <= 11.5 (True)
    # 10.5 <= 12.0 <= 12.5 (True)
    # 13.0 <= 15.0 <= 15.0 (True)
    # 14.0 <= 14.0 <= 16.0 (True)
    # 16.0 <= 18.0 <= 18.0 (True)
    assert result.diagnostics["coverage"] == 1.0


def test_calibrate_prediction_intervals_holdout():
    """Verify holdout indices expand the interval width appropriately."""
    time = [1.0, 2.0, 3.0]
    observed = [10.0, 10.0, 15.0]
    predicted = [10.0, 10.0, 12.0]

    # Residuals: [0.0, 0.0, 3.0]
    # Absolute sorted: [0.0, 0.0, 3.0]
    # Confidence 0.5: round(0.5 * 2) = 1 -> half_width = 0.0

    # But if index 2 is holdout, holdout width is 3.0.
    # So max(0.0, 3.0) = 3.0 -> half_width = 3.0
    holdout = [0, 0, 1]

    result = calibrate_prediction_intervals(
        time=time, observed=observed, predicted=predicted, confidence=0.5, holdout=holdout
    )

    assert result.metadata["interval_half_width"] == 3.0
    assert result.lower == [7.0, 7.0, 9.0]
    assert result.upper == [13.0, 13.0, 15.0]


def test_calibrate_prediction_intervals_value_errors():
    """Verify ValueErrors for invalid inputs."""
    time = [1.0, 2.0, 3.0]
    observed = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 3.0]

    # Invalid confidence
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        calibrate_prediction_intervals(time=time, observed=observed, predicted=predicted, confidence=0.0)

    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        calibrate_prediction_intervals(time=time, observed=observed, predicted=predicted, confidence=1.0)

    # Mismatched lengths
    with pytest.raises(ValueError, match="observed and predicted lengths must match time length"):
        calibrate_prediction_intervals(time=time, observed=[1.0, 2.0], predicted=predicted)

    with pytest.raises(ValueError, match="observed and predicted lengths must match time length"):
        calibrate_prediction_intervals(time=time, observed=observed, predicted=[1.0, 2.0])

    with pytest.raises(ValueError, match="holdout length must match time length"):
        calibrate_prediction_intervals(time=time, observed=observed, predicted=predicted, holdout=[1])
