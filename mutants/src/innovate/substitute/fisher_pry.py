# src/innovate/substitute/fisher_pry.py

from collections.abc import Sequence

import numpy as np

from innovate import backend
from innovate.base.base import DiffusionModel
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


class FisherPryModel(DiffusionModel):
    """Implementation of the Fisher-Pry model for technology substitution.

    This model assumes that the substitution of a new technology for an old one
    follows a logistic growth curve. The model tracks the market share
    fraction of the new technology.
    """

    def __init__(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁFisherPryModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁFisherPryModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁFisherPryModelǁ__init____mutmut_orig(self):
        self._params: dict[str, float] = {}

    def xǁFisherPryModelǁ__init____mutmut_1(self):
        self._params: dict[str, float] = None
    
    xǁFisherPryModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁFisherPryModelǁ__init____mutmut_1': xǁFisherPryModelǁ__init____mutmut_1
    }
    xǁFisherPryModelǁ__init____mutmut_orig.__name__ = 'xǁFisherPryModelǁ__init__'

    @property
    def param_names(self) -> Sequence[str]:
        """Returns the names of the model parameters: alpha and t0."""
        return ["alpha", "t0"]

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁFisherPryModelǁinitial_guesses__mutmut_orig'), object.__getattribute__(self, 'xǁFisherPryModelǁinitial_guesses__mutmut_mutants'), args, kwargs, self)

    def xǁFisherPryModelǁinitial_guesses__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = None
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(None)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = None

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(None)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = None

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(None)]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(None))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr + 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 1.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = None
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(None, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, None, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, None)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, )
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1.000001, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 + 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 2 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1.000001)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = None

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(None)

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped * (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 + y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (2 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = None
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(None, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, None, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, None)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_30(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_31(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_32(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, )
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_33(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 2)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_34(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = None  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_35(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(None, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_36(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, None)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_37(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_38(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, )  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_39(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(1, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_40(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = None  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_41(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 1.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_42(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "XXalphaXX": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_43(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "ALPHA": alpha_guess,
            "t0": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_44(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "XXt0XX": t0_guess,
        }

    def xǁFisherPryModelǁinitial_guesses__mutmut_45(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = backend.current_backend.array(y)
        t_arr = backend.current_backend.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[backend.current_backend.argmin(backend.current_backend.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = backend.current_backend.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = backend.current_backend.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope)  # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5  # Fallback value

        return {
            "alpha": alpha_guess,
            "T0": t0_guess,
        }
    
    xǁFisherPryModelǁinitial_guesses__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁFisherPryModelǁinitial_guesses__mutmut_1': xǁFisherPryModelǁinitial_guesses__mutmut_1, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_2': xǁFisherPryModelǁinitial_guesses__mutmut_2, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_3': xǁFisherPryModelǁinitial_guesses__mutmut_3, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_4': xǁFisherPryModelǁinitial_guesses__mutmut_4, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_5': xǁFisherPryModelǁinitial_guesses__mutmut_5, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_6': xǁFisherPryModelǁinitial_guesses__mutmut_6, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_7': xǁFisherPryModelǁinitial_guesses__mutmut_7, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_8': xǁFisherPryModelǁinitial_guesses__mutmut_8, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_9': xǁFisherPryModelǁinitial_guesses__mutmut_9, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_10': xǁFisherPryModelǁinitial_guesses__mutmut_10, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_11': xǁFisherPryModelǁinitial_guesses__mutmut_11, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_12': xǁFisherPryModelǁinitial_guesses__mutmut_12, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_13': xǁFisherPryModelǁinitial_guesses__mutmut_13, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_14': xǁFisherPryModelǁinitial_guesses__mutmut_14, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_15': xǁFisherPryModelǁinitial_guesses__mutmut_15, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_16': xǁFisherPryModelǁinitial_guesses__mutmut_16, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_17': xǁFisherPryModelǁinitial_guesses__mutmut_17, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_18': xǁFisherPryModelǁinitial_guesses__mutmut_18, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_19': xǁFisherPryModelǁinitial_guesses__mutmut_19, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_20': xǁFisherPryModelǁinitial_guesses__mutmut_20, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_21': xǁFisherPryModelǁinitial_guesses__mutmut_21, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_22': xǁFisherPryModelǁinitial_guesses__mutmut_22, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_23': xǁFisherPryModelǁinitial_guesses__mutmut_23, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_24': xǁFisherPryModelǁinitial_guesses__mutmut_24, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_25': xǁFisherPryModelǁinitial_guesses__mutmut_25, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_26': xǁFisherPryModelǁinitial_guesses__mutmut_26, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_27': xǁFisherPryModelǁinitial_guesses__mutmut_27, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_28': xǁFisherPryModelǁinitial_guesses__mutmut_28, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_29': xǁFisherPryModelǁinitial_guesses__mutmut_29, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_30': xǁFisherPryModelǁinitial_guesses__mutmut_30, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_31': xǁFisherPryModelǁinitial_guesses__mutmut_31, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_32': xǁFisherPryModelǁinitial_guesses__mutmut_32, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_33': xǁFisherPryModelǁinitial_guesses__mutmut_33, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_34': xǁFisherPryModelǁinitial_guesses__mutmut_34, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_35': xǁFisherPryModelǁinitial_guesses__mutmut_35, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_36': xǁFisherPryModelǁinitial_guesses__mutmut_36, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_37': xǁFisherPryModelǁinitial_guesses__mutmut_37, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_38': xǁFisherPryModelǁinitial_guesses__mutmut_38, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_39': xǁFisherPryModelǁinitial_guesses__mutmut_39, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_40': xǁFisherPryModelǁinitial_guesses__mutmut_40, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_41': xǁFisherPryModelǁinitial_guesses__mutmut_41, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_42': xǁFisherPryModelǁinitial_guesses__mutmut_42, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_43': xǁFisherPryModelǁinitial_guesses__mutmut_43, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_44': xǁFisherPryModelǁinitial_guesses__mutmut_44, 
        'xǁFisherPryModelǁinitial_guesses__mutmut_45': xǁFisherPryModelǁinitial_guesses__mutmut_45
    }
    xǁFisherPryModelǁinitial_guesses__mutmut_orig.__name__ = 'xǁFisherPryModelǁinitial_guesses'

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁFisherPryModelǁbounds__mutmut_orig'), object.__getattribute__(self, 'xǁFisherPryModelǁbounds__mutmut_mutants'), args, kwargs, self)

    def xǁFisherPryModelǁbounds__mutmut_orig(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(t)
        t_range = t_max - t_min
        return {
            "alpha": (0, np.inf),
            "t0": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_1(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = None
        t_range = t_max - t_min
        return {
            "alpha": (0, np.inf),
            "t0": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_2(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(None), backend.current_backend.max(t)
        t_range = t_max - t_min
        return {
            "alpha": (0, np.inf),
            "t0": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_3(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(None)
        t_range = t_max - t_min
        return {
            "alpha": (0, np.inf),
            "t0": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_4(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(t)
        t_range = None
        return {
            "alpha": (0, np.inf),
            "t0": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_5(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(t)
        t_range = t_max + t_min
        return {
            "alpha": (0, np.inf),
            "t0": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_6(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(t)
        t_range = t_max - t_min
        return {
            "XXalphaXX": (0, np.inf),
            "t0": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_7(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(t)
        t_range = t_max - t_min
        return {
            "ALPHA": (0, np.inf),
            "t0": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_8(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(t)
        t_range = t_max - t_min
        return {
            "alpha": (1, np.inf),
            "t0": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_9(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(t)
        t_range = t_max - t_min
        return {
            "alpha": (0, np.inf),
            "XXt0XX": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_10(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(t)
        t_range = t_max - t_min
        return {
            "alpha": (0, np.inf),
            "T0": (t_min - t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_11(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(t)
        t_range = t_max - t_min
        return {
            "alpha": (0, np.inf),
            "t0": (t_min + t_range, t_max + t_range),
        }

    def xǁFisherPryModelǁbounds__mutmut_12(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = backend.current_backend.min(t), backend.current_backend.max(t)
        t_range = t_max - t_min
        return {
            "alpha": (0, np.inf),
            "t0": (t_min - t_range, t_max - t_range),
        }
    
    xǁFisherPryModelǁbounds__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁFisherPryModelǁbounds__mutmut_1': xǁFisherPryModelǁbounds__mutmut_1, 
        'xǁFisherPryModelǁbounds__mutmut_2': xǁFisherPryModelǁbounds__mutmut_2, 
        'xǁFisherPryModelǁbounds__mutmut_3': xǁFisherPryModelǁbounds__mutmut_3, 
        'xǁFisherPryModelǁbounds__mutmut_4': xǁFisherPryModelǁbounds__mutmut_4, 
        'xǁFisherPryModelǁbounds__mutmut_5': xǁFisherPryModelǁbounds__mutmut_5, 
        'xǁFisherPryModelǁbounds__mutmut_6': xǁFisherPryModelǁbounds__mutmut_6, 
        'xǁFisherPryModelǁbounds__mutmut_7': xǁFisherPryModelǁbounds__mutmut_7, 
        'xǁFisherPryModelǁbounds__mutmut_8': xǁFisherPryModelǁbounds__mutmut_8, 
        'xǁFisherPryModelǁbounds__mutmut_9': xǁFisherPryModelǁbounds__mutmut_9, 
        'xǁFisherPryModelǁbounds__mutmut_10': xǁFisherPryModelǁbounds__mutmut_10, 
        'xǁFisherPryModelǁbounds__mutmut_11': xǁFisherPryModelǁbounds__mutmut_11, 
        'xǁFisherPryModelǁbounds__mutmut_12': xǁFisherPryModelǁbounds__mutmut_12
    }
    xǁFisherPryModelǁbounds__mutmut_orig.__name__ = 'xǁFisherPryModelǁbounds'

    def differential_equation(self, y, t, alpha, t0):
        args = [y, t, alpha, t0]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁFisherPryModelǁdifferential_equation__mutmut_orig'), object.__getattribute__(self, 'xǁFisherPryModelǁdifferential_equation__mutmut_mutants'), args, kwargs, self)

    def xǁFisherPryModelǁdifferential_equation__mutmut_orig(self, y, t, alpha, t0):
        """The differential equation for the Fisher-Pry model."""
        return alpha * y * (1 - y)

    def xǁFisherPryModelǁdifferential_equation__mutmut_1(self, y, t, alpha, t0):
        """The differential equation for the Fisher-Pry model."""
        return alpha * y / (1 - y)

    def xǁFisherPryModelǁdifferential_equation__mutmut_2(self, y, t, alpha, t0):
        """The differential equation for the Fisher-Pry model."""
        return alpha / y * (1 - y)

    def xǁFisherPryModelǁdifferential_equation__mutmut_3(self, y, t, alpha, t0):
        """The differential equation for the Fisher-Pry model."""
        return alpha * y * (1 + y)

    def xǁFisherPryModelǁdifferential_equation__mutmut_4(self, y, t, alpha, t0):
        """The differential equation for the Fisher-Pry model."""
        return alpha * y * (2 - y)
    
    xǁFisherPryModelǁdifferential_equation__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁFisherPryModelǁdifferential_equation__mutmut_1': xǁFisherPryModelǁdifferential_equation__mutmut_1, 
        'xǁFisherPryModelǁdifferential_equation__mutmut_2': xǁFisherPryModelǁdifferential_equation__mutmut_2, 
        'xǁFisherPryModelǁdifferential_equation__mutmut_3': xǁFisherPryModelǁdifferential_equation__mutmut_3, 
        'xǁFisherPryModelǁdifferential_equation__mutmut_4': xǁFisherPryModelǁdifferential_equation__mutmut_4
    }
    xǁFisherPryModelǁdifferential_equation__mutmut_orig.__name__ = 'xǁFisherPryModelǁdifferential_equation'

    def predict(self, t: Sequence[float]) -> Sequence[float]:
        args = [t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁFisherPryModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁFisherPryModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁFisherPryModelǁpredict__mutmut_orig(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_1(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_2(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError(None)

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_3(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_4(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_5(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_6(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = None
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_7(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(None)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_8(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = None
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_9(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["XXalphaXX"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_10(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["ALPHA"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_11(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = None
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_12(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["XXt0XX"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_13(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["T0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_14(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 * (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_15(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 2 / (1 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_16(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 - backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_17(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (2 + backend.current_backend.exp(-alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_18(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(None))

    def xǁFisherPryModelǁpredict__mutmut_19(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha / (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_20(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(+alpha * (t_arr - t0)))

    def xǁFisherPryModelǁpredict__mutmut_21(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the market share fraction of the new technology.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)
        alpha = self._params["alpha"]
        t0 = self._params["t0"]
        return 1 / (1 + backend.current_backend.exp(-alpha * (t_arr + t0)))
    
    xǁFisherPryModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁFisherPryModelǁpredict__mutmut_1': xǁFisherPryModelǁpredict__mutmut_1, 
        'xǁFisherPryModelǁpredict__mutmut_2': xǁFisherPryModelǁpredict__mutmut_2, 
        'xǁFisherPryModelǁpredict__mutmut_3': xǁFisherPryModelǁpredict__mutmut_3, 
        'xǁFisherPryModelǁpredict__mutmut_4': xǁFisherPryModelǁpredict__mutmut_4, 
        'xǁFisherPryModelǁpredict__mutmut_5': xǁFisherPryModelǁpredict__mutmut_5, 
        'xǁFisherPryModelǁpredict__mutmut_6': xǁFisherPryModelǁpredict__mutmut_6, 
        'xǁFisherPryModelǁpredict__mutmut_7': xǁFisherPryModelǁpredict__mutmut_7, 
        'xǁFisherPryModelǁpredict__mutmut_8': xǁFisherPryModelǁpredict__mutmut_8, 
        'xǁFisherPryModelǁpredict__mutmut_9': xǁFisherPryModelǁpredict__mutmut_9, 
        'xǁFisherPryModelǁpredict__mutmut_10': xǁFisherPryModelǁpredict__mutmut_10, 
        'xǁFisherPryModelǁpredict__mutmut_11': xǁFisherPryModelǁpredict__mutmut_11, 
        'xǁFisherPryModelǁpredict__mutmut_12': xǁFisherPryModelǁpredict__mutmut_12, 
        'xǁFisherPryModelǁpredict__mutmut_13': xǁFisherPryModelǁpredict__mutmut_13, 
        'xǁFisherPryModelǁpredict__mutmut_14': xǁFisherPryModelǁpredict__mutmut_14, 
        'xǁFisherPryModelǁpredict__mutmut_15': xǁFisherPryModelǁpredict__mutmut_15, 
        'xǁFisherPryModelǁpredict__mutmut_16': xǁFisherPryModelǁpredict__mutmut_16, 
        'xǁFisherPryModelǁpredict__mutmut_17': xǁFisherPryModelǁpredict__mutmut_17, 
        'xǁFisherPryModelǁpredict__mutmut_18': xǁFisherPryModelǁpredict__mutmut_18, 
        'xǁFisherPryModelǁpredict__mutmut_19': xǁFisherPryModelǁpredict__mutmut_19, 
        'xǁFisherPryModelǁpredict__mutmut_20': xǁFisherPryModelǁpredict__mutmut_20, 
        'xǁFisherPryModelǁpredict__mutmut_21': xǁFisherPryModelǁpredict__mutmut_21
    }
    xǁFisherPryModelǁpredict__mutmut_orig.__name__ = 'xǁFisherPryModelǁpredict'

    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁFisherPryModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁFisherPryModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁFisherPryModelǁscore__mutmut_orig(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_1(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_2(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError(None)
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_3(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_4(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_5(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_6(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = None
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_7(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(None)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_8(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = None
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_9(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            None,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_10(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) * 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_11(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) + y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_12(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(None) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_13(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 3,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_14(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = None
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_15(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            None,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_16(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) * 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_17(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) + backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_18(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(None) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_19(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(None)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_20(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(None))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_21(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 3,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_22(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_23(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_24(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_25(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0

    def xǁFisherPryModelǁscore__mutmut_26(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0

    def xǁFisherPryModelǁscore__mutmut_27(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Calculates the R^2 score for the model fit."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(backend.current_backend.array(y))) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    xǁFisherPryModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁFisherPryModelǁscore__mutmut_1': xǁFisherPryModelǁscore__mutmut_1, 
        'xǁFisherPryModelǁscore__mutmut_2': xǁFisherPryModelǁscore__mutmut_2, 
        'xǁFisherPryModelǁscore__mutmut_3': xǁFisherPryModelǁscore__mutmut_3, 
        'xǁFisherPryModelǁscore__mutmut_4': xǁFisherPryModelǁscore__mutmut_4, 
        'xǁFisherPryModelǁscore__mutmut_5': xǁFisherPryModelǁscore__mutmut_5, 
        'xǁFisherPryModelǁscore__mutmut_6': xǁFisherPryModelǁscore__mutmut_6, 
        'xǁFisherPryModelǁscore__mutmut_7': xǁFisherPryModelǁscore__mutmut_7, 
        'xǁFisherPryModelǁscore__mutmut_8': xǁFisherPryModelǁscore__mutmut_8, 
        'xǁFisherPryModelǁscore__mutmut_9': xǁFisherPryModelǁscore__mutmut_9, 
        'xǁFisherPryModelǁscore__mutmut_10': xǁFisherPryModelǁscore__mutmut_10, 
        'xǁFisherPryModelǁscore__mutmut_11': xǁFisherPryModelǁscore__mutmut_11, 
        'xǁFisherPryModelǁscore__mutmut_12': xǁFisherPryModelǁscore__mutmut_12, 
        'xǁFisherPryModelǁscore__mutmut_13': xǁFisherPryModelǁscore__mutmut_13, 
        'xǁFisherPryModelǁscore__mutmut_14': xǁFisherPryModelǁscore__mutmut_14, 
        'xǁFisherPryModelǁscore__mutmut_15': xǁFisherPryModelǁscore__mutmut_15, 
        'xǁFisherPryModelǁscore__mutmut_16': xǁFisherPryModelǁscore__mutmut_16, 
        'xǁFisherPryModelǁscore__mutmut_17': xǁFisherPryModelǁscore__mutmut_17, 
        'xǁFisherPryModelǁscore__mutmut_18': xǁFisherPryModelǁscore__mutmut_18, 
        'xǁFisherPryModelǁscore__mutmut_19': xǁFisherPryModelǁscore__mutmut_19, 
        'xǁFisherPryModelǁscore__mutmut_20': xǁFisherPryModelǁscore__mutmut_20, 
        'xǁFisherPryModelǁscore__mutmut_21': xǁFisherPryModelǁscore__mutmut_21, 
        'xǁFisherPryModelǁscore__mutmut_22': xǁFisherPryModelǁscore__mutmut_22, 
        'xǁFisherPryModelǁscore__mutmut_23': xǁFisherPryModelǁscore__mutmut_23, 
        'xǁFisherPryModelǁscore__mutmut_24': xǁFisherPryModelǁscore__mutmut_24, 
        'xǁFisherPryModelǁscore__mutmut_25': xǁFisherPryModelǁscore__mutmut_25, 
        'xǁFisherPryModelǁscore__mutmut_26': xǁFisherPryModelǁscore__mutmut_26, 
        'xǁFisherPryModelǁscore__mutmut_27': xǁFisherPryModelǁscore__mutmut_27
    }
    xǁFisherPryModelǁscore__mutmut_orig.__name__ = 'xǁFisherPryModelǁscore'

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = value

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        args = [t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁFisherPryModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁFisherPryModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_orig(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_1(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_2(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError(None)
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_3(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_4(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_5(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_6(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = None
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_7(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = backend.current_backend.array(None)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_8(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = backend.current_backend.array(t)
        y_pred = None
        return self.differential_equation(y_pred, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_9(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(None)
        return self.differential_equation(y_pred, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_10(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(None, t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_11(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, None, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_12(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(t_arr, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_13(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, **self._params)

    def xǁFisherPryModelǁpredict_adoption_rate__mutmut_14(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = backend.current_backend.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, t_arr, )
    
    xǁFisherPryModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁFisherPryModelǁpredict_adoption_rate__mutmut_1': xǁFisherPryModelǁpredict_adoption_rate__mutmut_1, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_2': xǁFisherPryModelǁpredict_adoption_rate__mutmut_2, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_3': xǁFisherPryModelǁpredict_adoption_rate__mutmut_3, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_4': xǁFisherPryModelǁpredict_adoption_rate__mutmut_4, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_5': xǁFisherPryModelǁpredict_adoption_rate__mutmut_5, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_6': xǁFisherPryModelǁpredict_adoption_rate__mutmut_6, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_7': xǁFisherPryModelǁpredict_adoption_rate__mutmut_7, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_8': xǁFisherPryModelǁpredict_adoption_rate__mutmut_8, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_9': xǁFisherPryModelǁpredict_adoption_rate__mutmut_9, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_10': xǁFisherPryModelǁpredict_adoption_rate__mutmut_10, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_11': xǁFisherPryModelǁpredict_adoption_rate__mutmut_11, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_12': xǁFisherPryModelǁpredict_adoption_rate__mutmut_12, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_13': xǁFisherPryModelǁpredict_adoption_rate__mutmut_13, 
        'xǁFisherPryModelǁpredict_adoption_rate__mutmut_14': xǁFisherPryModelǁpredict_adoption_rate__mutmut_14
    }
    xǁFisherPryModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁFisherPryModelǁpredict_adoption_rate'

    def fit(self, fitter, t: Sequence[float], y: Sequence[float], **kwargs):
        args = [fitter, t, y]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁFisherPryModelǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁFisherPryModelǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁFisherPryModelǁfit__mutmut_orig(self, fitter, t: Sequence[float], y: Sequence[float], **kwargs):
        """Fits the Fisher-Pry model to the data.

        Note: The input `y` for the Fisher-Pry model should be the market
        share fraction (between 0 and 1) of the new technology.
        """
        return super().fit(fitter, t, y, **kwargs)

    def xǁFisherPryModelǁfit__mutmut_1(self, fitter, t: Sequence[float], y: Sequence[float], **kwargs):
        """Fits the Fisher-Pry model to the data.

        Note: The input `y` for the Fisher-Pry model should be the market
        share fraction (between 0 and 1) of the new technology.
        """
        return super().fit(None, t, y, **kwargs)

    def xǁFisherPryModelǁfit__mutmut_2(self, fitter, t: Sequence[float], y: Sequence[float], **kwargs):
        """Fits the Fisher-Pry model to the data.

        Note: The input `y` for the Fisher-Pry model should be the market
        share fraction (between 0 and 1) of the new technology.
        """
        return super().fit(fitter, None, y, **kwargs)

    def xǁFisherPryModelǁfit__mutmut_3(self, fitter, t: Sequence[float], y: Sequence[float], **kwargs):
        """Fits the Fisher-Pry model to the data.

        Note: The input `y` for the Fisher-Pry model should be the market
        share fraction (between 0 and 1) of the new technology.
        """
        return super().fit(fitter, t, None, **kwargs)

    def xǁFisherPryModelǁfit__mutmut_4(self, fitter, t: Sequence[float], y: Sequence[float], **kwargs):
        """Fits the Fisher-Pry model to the data.

        Note: The input `y` for the Fisher-Pry model should be the market
        share fraction (between 0 and 1) of the new technology.
        """
        return super().fit(t, y, **kwargs)

    def xǁFisherPryModelǁfit__mutmut_5(self, fitter, t: Sequence[float], y: Sequence[float], **kwargs):
        """Fits the Fisher-Pry model to the data.

        Note: The input `y` for the Fisher-Pry model should be the market
        share fraction (between 0 and 1) of the new technology.
        """
        return super().fit(fitter, y, **kwargs)

    def xǁFisherPryModelǁfit__mutmut_6(self, fitter, t: Sequence[float], y: Sequence[float], **kwargs):
        """Fits the Fisher-Pry model to the data.

        Note: The input `y` for the Fisher-Pry model should be the market
        share fraction (between 0 and 1) of the new technology.
        """
        return super().fit(fitter, t, **kwargs)

    def xǁFisherPryModelǁfit__mutmut_7(self, fitter, t: Sequence[float], y: Sequence[float], **kwargs):
        """Fits the Fisher-Pry model to the data.

        Note: The input `y` for the Fisher-Pry model should be the market
        share fraction (between 0 and 1) of the new technology.
        """
        return super().fit(fitter, t, y, )
    
    xǁFisherPryModelǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁFisherPryModelǁfit__mutmut_1': xǁFisherPryModelǁfit__mutmut_1, 
        'xǁFisherPryModelǁfit__mutmut_2': xǁFisherPryModelǁfit__mutmut_2, 
        'xǁFisherPryModelǁfit__mutmut_3': xǁFisherPryModelǁfit__mutmut_3, 
        'xǁFisherPryModelǁfit__mutmut_4': xǁFisherPryModelǁfit__mutmut_4, 
        'xǁFisherPryModelǁfit__mutmut_5': xǁFisherPryModelǁfit__mutmut_5, 
        'xǁFisherPryModelǁfit__mutmut_6': xǁFisherPryModelǁfit__mutmut_6, 
        'xǁFisherPryModelǁfit__mutmut_7': xǁFisherPryModelǁfit__mutmut_7
    }
    xǁFisherPryModelǁfit__mutmut_orig.__name__ = 'xǁFisherPryModelǁfit'
