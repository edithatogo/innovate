from abc import ABC, abstractmethod
from typing import Dict, Sequence, TypeVar, Any

Self = TypeVar('Self')

class SystemBehavior(ABC):
    """Abstract base class for all system behavior models."""

    @abstractmethod
    def compute_behavior_rates(self, **params):
        """Calculates the instantaneous behavior rates."""
        pass

    @abstractmethod
    def predict_states(self, time_points, **params):
        """Predicts the states of the system over time."""
        pass

    @abstractmethod
    def get_parameters_schema(self):
        """Returns the schema for the model's parameters."""
        pass
