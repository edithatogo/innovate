from abc import ABC, abstractmethod
from typing import TypeVar

Self = TypeVar("Self")
from typing import Annotated
from typing import Callable
from typing import ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"] # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None): # type: ignore
    """Forward call to original or mutated function, depending on the environment"""
    import os # type: ignore
    mutant_under_test = os.environ['MUTANT_UNDER_TEST'] # type: ignore
    if mutant_under_test == 'fail': # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException # type: ignore
        raise MutmutProgrammaticFailException('Failed programmatically')       # type: ignore
    elif mutant_under_test == 'stats': # type: ignore
        from mutmut.__main__ import record_trampoline_hit # type: ignore
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__) # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_' # type: ignore
    if not mutant_under_test.startswith(prefix): # type: ignore
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    mutant_name = mutant_under_test.rpartition('.')[-1] # type: ignore
    if self_arg is not None: # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs) # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs) # type: ignore
    return result # type: ignore


class ContagionSpread(ABC):
    """Abstract base class for all contagion spread models."""

    @abstractmethod
    def compute_spread_rate(self, **params):
        """Calculates the instantaneous spread rate."""
        """
        Calculate the instantaneous rate at which contagion spreads based on provided model parameters.

        Parameters:
        	params: Arbitrary keyword arguments representing model-specific parameters required to compute the spread rate.

        Returns:
        	The computed spread rate, as defined by the specific contagion model.
        """

    @abstractmethod
    def predict_states(self, time_points, **params):
        """Predicts the states of the population over time."""
        """
        Predict the states of the population at specified time points using the given model parameters.

        Parameters:
            time_points (Iterable): Sequence of time points at which to predict population states.
            **params: Model-specific parameters required for prediction.

        Returns:
            Any: Predicted population states at each specified time point.
        """

    @abstractmethod
    def get_parameters_schema(self):
        """Returns the schema for the model's parameters.

        Return the schema describing the parameters required by the contagion spread model.

        Returns
        -------
            dict: A schema detailing the expected parameters for the model.
        """
