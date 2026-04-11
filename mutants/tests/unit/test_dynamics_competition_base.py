"""Tests for the competition dynamics base module."""

import pytest

from src.innovate.dynamics.competition.base import CompetitiveInteraction


class MockCompetitiveInteraction(CompetitiveInteraction):
    """Mock implementation of CompetitiveInteraction for testing."""

    def compute_interaction_rates(self, **params):
        """Mock implementation of compute_interaction_rates."""
        return params.get("rate", 1.0)

    def predict_states(self, time_points, **params):
        """Mock implementation of predict_states."""
        return [time * params.get("rate", 1.0) for time in time_points]

    def get_parameters_schema(self):
        """Mock implementation of get_parameters_schema."""
        return {"rate": "float", "interactions": "int"}


def test_competitive_interaction_abstract_base_class():
    """Test that CompetitiveInteraction is an abstract base class."""
    # Attempting to instantiate CompetitiveInteraction directly should raise TypeError
    with pytest.raises(TypeError):
        CompetitiveInteraction()

    # But we can instantiate a concrete implementation
    mock_interaction = MockCompetitiveInteraction()
    assert mock_interaction.compute_interaction_rates() == 1.0


def test_competitive_interaction_compute_interaction_rates():
    """Test the compute_interaction_rates method of a concrete implementation."""
    mock_interaction = MockCompetitiveInteraction()

    # Test with default parameters
    assert mock_interaction.compute_interaction_rates() == 1.0

    # Test with custom parameters
    assert mock_interaction.compute_interaction_rates(rate=2.0) == 2.0
    assert mock_interaction.compute_interaction_rates(rate=0.5) == 0.5
    assert mock_interaction.compute_interaction_rates(rate=5.0, other_param="ignored") == 5.0


def test_competitive_interaction_predict_states():
    """Test the predict_states method of a concrete implementation."""
    mock_interaction = MockCompetitiveInteraction()

    # Test with default rate
    time_points = [0, 1, 2, 3]
    states = mock_interaction.predict_states(time_points)
    expected = [0, 1, 2, 3]
    assert states == expected

    # Test with custom rate
    states = mock_interaction.predict_states(time_points, rate=2.0)
    expected = [0, 2, 4, 6]
    assert states == expected

    # Test with different time points
    time_points = [1, 3, 5]
    states = mock_interaction.predict_states(time_points, rate=0.5)
    expected = [0.5, 1.5, 2.5]
    assert states == expected


def test_competitive_interaction_get_parameters_schema():
    """Test the get_parameters_schema method of a concrete implementation."""
    mock_interaction = MockCompetitiveInteraction()

    schema = mock_interaction.get_parameters_schema()
    expected_schema = {"rate": "float", "interactions": "int"}
    assert schema == expected_schema
