from abc import ABC, abstractmethod
from typing import Dict, Sequence, TypeVar, Any

Self = TypeVar('Self')

class GrowthCurve(ABC):
    """Abstract base class for all growth curve models."""

    @abstractmethod
    def compute_growth_rate(self, current_adopters, total_potential, **params):
        """
        Calculates the instantaneous growth rate.

        Calculate the instantaneous growth rate based on the current number of adopters, total potential adopters, and additional model parameters.
        
        Parameters:
            current_adopters: The current number of adopters.
            total_potential: The total number of potential adopters.
            **params: Additional parameters specific to the growth model.
        
        Returns:
            The instantaneous growth rate as determined by the model.
        """
        pass

    @abstractmethod
    def predict_cumulative(self, time_points, initial_adopters, total_potential, **params):
        """
        Predicts cumulative adopters over time.

        Predict the cumulative number of adopters at specified time points.
        
        Parameters:
            time_points (Sequence[float]): Time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time.
            total_potential (float): Total potential number of adopters.
            **params: Additional model-specific parameters.
        
        Returns:
            Sequence[float]: Predicted cumulative adopters at each time point.
        """
        pass

    @abstractmethod
    def get_parameters_schema(self):

        """
        Returns the schema for the model's parameters.

        Return the schema describing the parameters required by the growth curve model.
        
        Returns:
            dict: A schema detailing the expected parameters for the model.
        """
        pass