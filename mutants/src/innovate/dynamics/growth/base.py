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


class GrowthCurve(ABC):
    """Abstract base class for all growth curve models."""

    @abstractmethod
    def compute_growth_rate(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Calculate the instantaneous growth rate based on the current number of adopters, total potential adopters, and additional model parameters.

        Parameters
        ----------
            current_adopters: The current number of adopters.
            total_potential: The total number of potential adopters.
            **params: Additional parameters specific to the growth model.

        Returns
        -------
            The instantaneous growth rate as determined by the model.
        """

    @abstractmethod
    def predict_cumulative(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predict the cumulative number of adopters at specified time points.

        Parameters
        ----------
            time_points (Sequence[float]): Time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time.
            total_potential (float): Total potential number of adopters.
            **params: Additional model-specific parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adopters at each time point.
        """

    @abstractmethod
    def get_parameters_schema(self):
        """Returns the schema for the model's parameters.

        Return the schema describing the parameters required by the growth curve model.

        Returns
        -------
            dict: A schema detailing the expected parameters for the model.
        """
