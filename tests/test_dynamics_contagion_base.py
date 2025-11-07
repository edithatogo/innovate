"""Tests for the contagion dynamics base module."""

import pytest

from src.innovate.dynamics.contagion.base import ContagionSpread


class MockContagionSpread(ContagionSpread):
    """Mock implementation of ContagionSpread for testing."""
    
    def compute_spread_rate(self, **params):
        """Mock implementation of compute_spread_rate."""
        return params.get("rate", 0.1)
    
    def predict_states(self, time_points, **params):
        """Mock implementation of predict_states."""
        rate = params.get("rate", 0.1)
        initial_state = params.get("initial_state", 0.0)
        return [initial_state + time * rate for time in time_points]
    
    def get_parameters_schema(self):
        """Mock implementation of get_parameters_schema."""
        return {"rate": "float", "initial_state": "float"}


def test_contagion_spread_abstract_base_class():
    """Test that ContagionSpread is an abstract base class."""
    # Attempting to instantiate ContagionSpread directly should raise TypeError
    with pytest.raises(TypeError):
        ContagionSpread()
    
    # But we can instantiate a concrete implementation
    mock_spread = MockContagionSpread()
    assert mock_spread.compute_spread_rate() == 0.1


def test_contagion_spread_compute_spread_rate():
    """Test the compute_spread_rate method of a concrete implementation."""
    mock_spread = MockContagionSpread()
    
    # Test with default parameters
    assert mock_spread.compute_spread_rate() == 0.1
    
    # Test with custom parameters
    assert mock_spread.compute_spread_rate(rate=0.5) == 0.5
    assert mock_spread.compute_spread_rate(rate=0.0) == 0.0
    assert mock_spread.compute_spread_rate(rate=2.0, other_param="ignored") == 2.0


def test_contagion_spread_predict_states():
    """Test the predict_states method of a concrete implementation."""
    mock_spread = MockContagionSpread()
    
    # Test with default parameters
    time_points = [0, 1, 2, 3]
    states = mock_spread.predict_states(time_points)
    expected = [0.0, 0.1, 0.2, 0.3]
    assert len(states) == len(expected)
    for s, e in zip(states, expected):
        assert abs(s - e) < 1e-10
    
    # Test with custom rate
    states = mock_spread.predict_states(time_points, rate=0.2)
    expected = [0.0, 0.2, 0.4, 0.6]
    assert len(states) == len(expected)
    for s, e in zip(states, expected):
        assert abs(s - e) < 1e-10
    
    # Test with custom initial state
    states = mock_spread.predict_states(time_points, initial_state=1.0, rate=0.1)
    expected = [1.0, 1.1, 1.2, 1.3]
    assert len(states) == len(expected)
    for s, e in zip(states, expected):
        assert abs(s - e) < 1e-10


def test_contagion_spread_get_parameters_schema():
    """Test the get_parameters_schema method of a concrete implementation."""
    mock_spread = MockContagionSpread()
    
    schema = mock_spread.get_parameters_schema()
    expected_schema = {"rate": "float", "initial_state": "float"}
    assert schema == expected_schema