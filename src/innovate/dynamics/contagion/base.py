from abc import ABC, abstractmethod
from typing import Dict, Sequence, TypeVar, Any

Self = TypeVar('Self')

class ContagionSpread(ABC):
    """Abstract base class for all contagion spread models."""

    @abstractmethod
    def compute_spread_rate(self, **params):
        """Calculates the instantaneous spread rate."""
        pass

    @abstractmethod
    def predict_states(self, time_points, **params):
        """Predicts the states of the population over time."""
        pass

    @abstractmethod
    def get_parameters_schema(self):
        """Returns the schema for the model's parameters."""
        pass
