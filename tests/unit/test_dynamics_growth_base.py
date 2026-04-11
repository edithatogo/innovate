"""Tests for the growth dynamics base module."""

import pytest

from src.innovate.dynamics.growth.base import GrowthCurve


class MockGrowthCurve(GrowthCurve):
    """Mock implementation of GrowthCurve for testing."""

    def compute_growth_rate(self, current_adopters, total_potential, **params):
        """Mock implementation of compute_growth_rate."""
        rate_param = params.get("rate", 1.0)
        return current_adopters * total_potential * rate_param

    def predict_cumulative(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Mock implementation of predict_cumulative."""
        rate_param = params.get("rate", 1.0)
        result = []
        for t in time_points:
            # Simple linear growth for testing purposes
            value = initial_adopters + t * rate_param
            # Keep within bounds of total potential
            value = min(value, total_potential)
            result.append(value)
        return result

    def get_parameters_schema(self):
        """Mock implementation of get_parameters_schema."""
        return {"rate": "float", "influence": "float"}


def test_growth_curve_abstract_base_class():
    """Test that GrowthCurve is an abstract base class."""
    # Attempting to instantiate GrowthCurve directly should raise TypeError
    with pytest.raises(TypeError):
        GrowthCurve()

    # But we can instantiate a concrete implementation
    mock_curve = MockGrowthCurve()
    assert mock_curve.compute_growth_rate(1, 10) == 10


def test_growth_curve_compute_growth_rate():
    """Test the compute_growth_rate method of a concrete implementation."""
    mock_curve = MockGrowthCurve()

    # Test with default parameters
    assert mock_curve.compute_growth_rate(0, 10) == 0
    assert mock_curve.compute_growth_rate(1, 10) == 10
    assert mock_curve.compute_growth_rate(2, 5) == 10

    # Test with custom rate parameter
    assert mock_curve.compute_growth_rate(1, 10, rate=2.0) == 20
    assert mock_curve.compute_growth_rate(3, 4, rate=0.5) == 6


def test_growth_curve_predict_cumulative():
    """Test the predict_cumulative method of a concrete implementation."""
    mock_curve = MockGrowthCurve()

    # Test with default parameters
    time_points = [0, 1, 2, 3]
    result = mock_curve.predict_cumulative(time_points, initial_adopters=1, total_potential=10)
    expected = [1, 2, 3, 4]  # Initial adopters (1) + time * rate (1)
    assert result == expected

    # Test with custom rate
    result = mock_curve.predict_cumulative(time_points, initial_adopters=2, total_potential=15, rate=1.5)
    expected = [2, 3.5, 5, 6.5]  # Initial adopters (2) + time * rate (1.5)
    assert result == expected

    # Test with values that should be capped by total potential
    result = mock_curve.predict_cumulative([0, 1, 2, 3, 4, 5], initial_adopters=1, total_potential=3, rate=1)
    expected = [1, 2, 3, 3, 3, 3]  # Should be capped at total potential of 3
    assert result == expected


def test_growth_curve_get_parameters_schema():
    """Test the get_parameters_schema method of a concrete implementation."""
    mock_curve = MockGrowthCurve()

    schema = mock_curve.get_parameters_schema()
    expected_schema = {"rate": "float", "influence": "float"}
    assert schema == expected_schema
