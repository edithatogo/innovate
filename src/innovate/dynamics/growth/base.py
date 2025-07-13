from abc import ABC, abstractmethod
from typing import Dict, Sequence, TypeVar, Any

Self = TypeVar('Self')

class GrowthCurve(ABC):
    """Abstract base class for all growth curve models."""

    @abstractmethod
    def compute_growth_rate(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate."""
        pass

    @abstractmethod
    def predict_cumulative(self, time_points, initial_adopters, total_potential, **params):
        """Predicts cumulative adopters over time."""
        pass

    @abstractmethod
    def get_parameters_schema(self):
        """Returns the schema for the model's parameters."""
        pass
