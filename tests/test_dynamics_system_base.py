"""Tests for the system dynamics base module."""

import pytest

from src.innovate.dynamics.system.base import SystemBehavior


class MockSystemBehavior(SystemBehavior):
    """Mock implementation of SystemBehavior for testing."""

    def compute_behavior_rates(self, **params):
        """Mock implementation of compute_behavior_rates."""
        rate_param = params.get("rate", 1.0)
        return rate_param

    def predict_states(self, time_points, **params):
        """Mock implementation of predict_states."""
        rate_param = params.get("rate", 1.0)
        initial_state = params.get("initial_state", 0.0)
        return [initial_state + t * rate_param for t in time_points]

    def get_parameters_schema(self):
        """Mock implementation of get_parameters_schema."""
        return {"rate": "float", "initial_state": "float", "threshold": "float"}


def test_system_behavior_abstract_base_class():
    """Test that SystemBehavior is an abstract base class."""
    # Attempting to instantiate SystemBehavior directly should raise TypeError
    with pytest.raises(TypeError):
        SystemBehavior()

    # But we can instantiate a concrete implementation
    mock_behavior = MockSystemBehavior()
    assert mock_behavior.compute_behavior_rates() == 1.0


def test_system_behavior_compute_behavior_rates():
    """Test the compute_behavior_rates method of a concrete implementation."""
    mock_behavior = MockSystemBehavior()

    # Test with default parameters
    assert mock_behavior.compute_behavior_rates() == 1.0

    # Test with custom parameters
    assert mock_behavior.compute_behavior_rates(rate=2.0) == 2.0
    assert mock_behavior.compute_behavior_rates(rate=0.5, other_param="ignored") == 0.5
    assert mock_behavior.compute_behavior_rates(rate=5.0) == 5.0


def test_system_behavior_predict_states():
    """Test the predict_states method of a concrete implementation."""
    mock_behavior = MockSystemBehavior()

    # Test with default parameters
    time_points = [0, 1, 2, 3]
    states = mock_behavior.predict_states(time_points)
    expected = [0.0, 1.0, 2.0, 3.0]  # initial_state=0 + time * rate=1
    assert states == expected

    # Test with custom rate
    states = mock_behavior.predict_states(time_points, rate=1.5)
    expected = [0.0, 1.5, 3.0, 4.5]  # initial_state=0 + time * rate=1.5
    assert states == expected

    # Test with custom initial state
    states = mock_behavior.predict_states(time_points, initial_state=5.0, rate=0.5)
    expected = [5.0, 5.5, 6.0, 6.5]  # initial_state=5 + time * rate=0.5
    assert states == expected

    # Test with single time point
    states = mock_behavior.predict_states([10], rate=2.0, initial_state=1.0)
    expected = [21.0]  # initial_state=1 + time=10 * rate=2
    assert states == expected


def test_system_behavior_get_parameters_schema():
    """Test the get_parameters_schema method of a concrete implementation."""
    mock_behavior = MockSystemBehavior()

    schema = mock_behavior.get_parameters_schema()
    expected_schema = {"rate": "float", "initial_state": "float", "threshold": "float"}
    assert schema == expected_schema
