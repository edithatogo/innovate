from abc import ABC, abstractmethod
from typing import Dict, Sequence, TypeVar, Any

Self = TypeVar('Self')

class CompetitiveInteraction(ABC):
    """Abstract base class for all competitive interaction models."""

    @abstractmethod
    def compute_interaction_rates(self, **params):
        """
        Calculate the instantaneous rates of interaction between competing entities based on provided parameters.
        
        Parameters:
        	params: Arbitrary keyword arguments representing model-specific parameters required to compute interaction rates.
        
        Returns:
        	Interaction rates as defined by the specific model implementation.
        """
        pass

    @abstractmethod
    def predict_states(self, time_points, **params):
        """
        Predict the states of competing entities at specified time points using provided parameters.
        
        Parameters:
            time_points: Sequence of time points at which to predict the states.
            **params: Model-specific parameters required for prediction.
        
        Returns:
            Predicted states of the competing entities at each specified time point.
        """
        pass

    @abstractmethod
    def get_parameters_schema(self):
        """
        Return the schema describing the parameters required by the competitive interaction model.
        
        Returns:
            The parameter schema, typically as a dictionary or structured object, defining expected model parameters.
        """
        pass
