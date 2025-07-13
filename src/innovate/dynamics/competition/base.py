from abc import ABC, abstractmethod
from typing import Dict, Sequence, TypeVar, Any

Self = TypeVar('Self')

class CompetitiveInteraction(ABC):
    """Abstract base class for all competitive interaction models."""

    @abstractmethod
    def compute_interaction_rates(self, **params):
        """Calculates the instantaneous interaction rates."""
        pass

    @abstractmethod
    def predict_states(self, time_points, **params):
        """Predicts the states of the competing entities over time."""
        pass

    @abstractmethod
    def get_parameters_schema(self):
        """Returns the schema for the model's parameters."""
        pass
