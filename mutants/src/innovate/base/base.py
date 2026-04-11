from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

# Define a type variable for the class itself, for type hinting Self
Self = TypeVar("Self", bound="DiffusionModel")
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


class DiffusionModel(ABC):
    """Abstract base class for all diffusion models."""

    @abstractmethod
    def predict(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts adoption levels for given time points.

        Parameters
        ----------
        t : Sequence[float]
            Sequence of time points for prediction

        Returns
        -------
        Sequence[float]
            Predicted adoption levels at each time point

        Raises
        ------
        RuntimeError
            If model is not fitted (params_ is empty)
        ValueError
            If time points are invalid (e.g., negative, non-numeric)
        """

    @abstractmethod
    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Returns the R^2 score of the model fit."""

    @property
    @abstractmethod
    def params_(self) -> dict[str, float]:
        """Returns a dictionary of fitted model parameters."""

    @params_.setter
    @abstractmethod
    def params_(self, value: dict[str, float]):
        """Sets the model parameters."""

    @abstractmethod
    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of adoption (new adoptions per unit of time)."""

    @property
    @abstractmethod
    def param_names(self) -> Sequence[str]:
        """Returns the names of the model parameters."""

    @abstractmethod
    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Returns initial guesses for the model parameters."""

    @abstractmethod
    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""

    @staticmethod
    @abstractmethod
    def differential_equation(y: float, t: float, p: Sequence[float]) -> float:
        """Returns the differential equation for the model."""

    def fit(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        args = [fitter, t, y]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDiffusionModelǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁDiffusionModelǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁDiffusionModelǁfit__mutmut_orig(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_1(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = None
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_2(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(None, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_3(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, None)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_4(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_5(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, )
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_6(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = None
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_7(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(None, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_8(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, None)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_9(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_10(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, )
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_11(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            None,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_12(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            None,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_13(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            None,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_14(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=None,
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_15(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=None,
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_16(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_17(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_18(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_19(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_20(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_21(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            )

    def xǁDiffusionModelǁfit__mutmut_22(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(None),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )

    def xǁDiffusionModelǁfit__mutmut_23(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(None),
            **kwargs,
        )
    
    xǁDiffusionModelǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDiffusionModelǁfit__mutmut_1': xǁDiffusionModelǁfit__mutmut_1, 
        'xǁDiffusionModelǁfit__mutmut_2': xǁDiffusionModelǁfit__mutmut_2, 
        'xǁDiffusionModelǁfit__mutmut_3': xǁDiffusionModelǁfit__mutmut_3, 
        'xǁDiffusionModelǁfit__mutmut_4': xǁDiffusionModelǁfit__mutmut_4, 
        'xǁDiffusionModelǁfit__mutmut_5': xǁDiffusionModelǁfit__mutmut_5, 
        'xǁDiffusionModelǁfit__mutmut_6': xǁDiffusionModelǁfit__mutmut_6, 
        'xǁDiffusionModelǁfit__mutmut_7': xǁDiffusionModelǁfit__mutmut_7, 
        'xǁDiffusionModelǁfit__mutmut_8': xǁDiffusionModelǁfit__mutmut_8, 
        'xǁDiffusionModelǁfit__mutmut_9': xǁDiffusionModelǁfit__mutmut_9, 
        'xǁDiffusionModelǁfit__mutmut_10': xǁDiffusionModelǁfit__mutmut_10, 
        'xǁDiffusionModelǁfit__mutmut_11': xǁDiffusionModelǁfit__mutmut_11, 
        'xǁDiffusionModelǁfit__mutmut_12': xǁDiffusionModelǁfit__mutmut_12, 
        'xǁDiffusionModelǁfit__mutmut_13': xǁDiffusionModelǁfit__mutmut_13, 
        'xǁDiffusionModelǁfit__mutmut_14': xǁDiffusionModelǁfit__mutmut_14, 
        'xǁDiffusionModelǁfit__mutmut_15': xǁDiffusionModelǁfit__mutmut_15, 
        'xǁDiffusionModelǁfit__mutmut_16': xǁDiffusionModelǁfit__mutmut_16, 
        'xǁDiffusionModelǁfit__mutmut_17': xǁDiffusionModelǁfit__mutmut_17, 
        'xǁDiffusionModelǁfit__mutmut_18': xǁDiffusionModelǁfit__mutmut_18, 
        'xǁDiffusionModelǁfit__mutmut_19': xǁDiffusionModelǁfit__mutmut_19, 
        'xǁDiffusionModelǁfit__mutmut_20': xǁDiffusionModelǁfit__mutmut_20, 
        'xǁDiffusionModelǁfit__mutmut_21': xǁDiffusionModelǁfit__mutmut_21, 
        'xǁDiffusionModelǁfit__mutmut_22': xǁDiffusionModelǁfit__mutmut_22, 
        'xǁDiffusionModelǁfit__mutmut_23': xǁDiffusionModelǁfit__mutmut_23
    }
    xǁDiffusionModelǁfit__mutmut_orig.__name__ = 'xǁDiffusionModelǁfit'
