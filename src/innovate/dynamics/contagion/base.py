from abc import ABC, abstractmethod
from typing import Dict, Sequence, TypeVar, Any

Self = TypeVar('Self')

class ContagionSpread(ABC):
    """Abstract base class for all contagion spread models."""

    @abstractmethod
    def compute_spread_rate(self, **params):
        """
        Calculate the instantaneous rate at which contagion spreads based on provided model parameters.
        
        Parameters:
        	params: Arbitrary keyword arguments representing model-specific parameters required to compute the spread rate.
        
        Returns:
        	The computed spread rate, as defined by the specific contagion model.
        """
        pass

    @abstractmethod
    def predict_states(self, time_points, **params):
        """
        Predict the states of the population at specified time points using the given model parameters.
        
        Parameters:
            time_points (Iterable): Sequence of time points at which to predict population states.
            **params: Model-specific parameters required for prediction.
        
        Returns:
            Any: Predicted population states at each specified time point.
        """
        pass

    @abstractmethod
    def get_parameters_schema(self):
        """
        Return the schema describing the parameters required by the contagion spread model.
        
        Returns:
            dict: A schema detailing the expected parameters for the model.
        """
        pass
