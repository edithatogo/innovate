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


class CompetitiveInteraction(ABC):
    """Abstract base class for all competitive interaction models."""

    @abstractmethod
    def compute_interaction_rates(self, **params):
        """Calculate the instantaneous rates of interaction between competing entities based on provided parameters.

        Parameters
        ----------
                params: Arbitrary keyword arguments representing model-specific parameters required to compute interaction rates.

        Returns
        -------
                Interaction rates as defined by the specific model implementation.
        """

    @abstractmethod
    def predict_states(self, time_points, **params):
        """Predict the states of competing entities at specified time points using provided parameters.

        Parameters
        ----------
            time_points: Sequence of time points at which to predict the states.
            **params: Model-specific parameters required for prediction.

        Returns
        -------
            Predicted states of the competing entities at each specified time point.
        """

    @abstractmethod
    def get_parameters_schema(self):
        """Return the schema describing the parameters required by the competitive interaction model.

        Returns
        -------
            The parameter schema, typically as a dictionary or structured object, defining expected model parameters.
        """
