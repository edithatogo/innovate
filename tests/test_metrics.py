# tests/test_metrics.py

import pytest
import numpy as np
from innovate.utils.metrics import calculate_rss, calculate_aic, calculate_bic


@pytest.fixture
def sample_data():
    """
    Provides sample data for testing metrics.
    """
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 9.5])
    # RSS = 0.5^2 * 9 + (-0.5)^2 = 0.25 * 9 + 0.25 = 2.25 + 0.25 = 2.5
    return y_true, y_pred


def test_calculate_rss(sample_data):
    """
    Tests the calculate_rss function.
    """
    y_true, y_pred = sample_data
    rss = calculate_rss(y_true, y_pred)
    assert np.isclose(rss, 2.5)


def test_calculate_aic(sample_data):
    """
    Tests the calculate_aic function.
    """
    y_true, y_pred = sample_data
    n_params = 2  # 1 for slope, 1 for variance of residuals
    n_samples = len(y_true)
    rss = calculate_rss(y_true, y_pred)

    # Manual calculation for verification
    log_likelihood = (
        -n_samples / 2 * np.log(2 * np.pi)
        - n_samples / 2 * np.log(rss / n_samples)
        - n_samples / 2
    )
    expected_aic = 2 * n_params - 2 * log_likelihood

    aic = calculate_aic(n_params, n_samples, rss)
    assert np.isclose(aic, expected_aic)
    # A known value for this case
    assert np.isclose(aic, 18.52, atol=0.01)


def test_calculate_bic(sample_data):
    """
    Tests the calculate_bic function.
    """
    y_true, y_pred = sample_data
    n_params = 2  # 1 for slope, 1 for variance of residuals
    n_samples = len(y_true)
    rss = calculate_rss(y_true, y_pred)

    # Manual calculation for verification
    log_likelihood = (
        -n_samples / 2 * np.log(2 * np.pi)
        - n_samples / 2 * np.log(rss / n_samples)
        - n_samples / 2
    )
    expected_bic = n_params * np.log(n_samples) - 2 * log_likelihood

    bic = calculate_bic(n_params, n_samples, rss)
    assert np.isclose(bic, expected_bic)
    # A known value for this case
    assert np.isclose(bic, 19.12, atol=0.01)
