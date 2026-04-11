# src/innovate/hype/hype_cycle.py

from collections.abc import Sequence

from numpy import array, clip, exp, inf, ndarray
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


class HypeCycleModel:
    """A model for generating a Hype Cycle curve.

    This model combines a logistic growth curve for the underlying technology
    maturity with a hype function to model the visibility of the technology.
    """

    def __init__(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHypeCycleModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁHypeCycleModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁHypeCycleModelǁ__init____mutmut_orig(self):
        self._params: dict[str, float] = {}

    def xǁHypeCycleModelǁ__init____mutmut_1(self):
        self._params: dict[str, float] = None
    
    xǁHypeCycleModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHypeCycleModelǁ__init____mutmut_1': xǁHypeCycleModelǁ__init____mutmut_1
    }
    xǁHypeCycleModelǁ__init____mutmut_orig.__name__ = 'xǁHypeCycleModelǁ__init__'

    @property
    def param_names(self) -> Sequence[str]:
        return [
            "k",  # Growth rate of the logistic curve
            "t0",  # Midpoint of the logistic curve
            "a_hype",  # Amplitude of the hype
            "t_hype",  # Time of the peak of the hype
            "w_hype",  # Width of the hype
            "a_d",  # Amplitude of the disillusionment
            "t_d",  # Time of the trough of disillusionment
            "w_d",  # Width of the disillusionment
        ]

    def predict(self, t: ndarray) -> ndarray:
        args = [t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHypeCycleModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁHypeCycleModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁHypeCycleModelǁpredict__mutmut_orig(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_1(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_2(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError(None)

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_3(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("XXModel parameters have not been set.XX")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_4(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_5(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("MODEL PARAMETERS HAVE NOT BEEN SET.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_6(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = None
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_7(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["XXkXX"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_8(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["K"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_9(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = None
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_10(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["XXt0XX"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_11(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["T0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_12(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = None
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_13(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["XXa_hypeXX"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_14(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["A_HYPE"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_15(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = None
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_16(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["XXt_hypeXX"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_17(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["T_HYPE"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_18(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = None
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_19(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["XXw_hypeXX"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_20(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["W_HYPE"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_21(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = None
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_22(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["XXa_dXX"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_23(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["A_D"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_24(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = None
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_25(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["XXt_dXX"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_26(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["T_D"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_27(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = None

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_28(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["XXw_dXX"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_29(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["W_D"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_30(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = None

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_31(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(None)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_32(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = None

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_33(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 * (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_34(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 1.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_35(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 - exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_36(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (2 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_37(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(None))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_38(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k / (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_39(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(+k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_40(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr + t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_41(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = None
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_42(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype / exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_43(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(None)
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_44(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) * (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_45(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(+((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_46(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) * 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_47(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr + t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_48(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 3) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_49(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 / w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_50(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (3 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_51(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype * 2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_52(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**3))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_53(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = None

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_54(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d / exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_55(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(None)

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_56(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) * (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_57(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(+((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_58(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) * 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_59(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr + t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_60(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 3) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_61(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 / w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_62(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (3 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_63(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d * 2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_64(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**3))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_65(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = None
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_66(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype + disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_67(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity - hype - disillusionment
        return clip(visibility, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_68(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(None, 0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_69(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, None, inf)

    def xǁHypeCycleModelǁpredict__mutmut_70(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, None)

    def xǁHypeCycleModelǁpredict__mutmut_71(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(0, inf)

    def xǁHypeCycleModelǁpredict__mutmut_72(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, inf)

    def xǁHypeCycleModelǁpredict__mutmut_73(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, )

    def xǁHypeCycleModelǁpredict__mutmut_74(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 1, inf)
    
    xǁHypeCycleModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHypeCycleModelǁpredict__mutmut_1': xǁHypeCycleModelǁpredict__mutmut_1, 
        'xǁHypeCycleModelǁpredict__mutmut_2': xǁHypeCycleModelǁpredict__mutmut_2, 
        'xǁHypeCycleModelǁpredict__mutmut_3': xǁHypeCycleModelǁpredict__mutmut_3, 
        'xǁHypeCycleModelǁpredict__mutmut_4': xǁHypeCycleModelǁpredict__mutmut_4, 
        'xǁHypeCycleModelǁpredict__mutmut_5': xǁHypeCycleModelǁpredict__mutmut_5, 
        'xǁHypeCycleModelǁpredict__mutmut_6': xǁHypeCycleModelǁpredict__mutmut_6, 
        'xǁHypeCycleModelǁpredict__mutmut_7': xǁHypeCycleModelǁpredict__mutmut_7, 
        'xǁHypeCycleModelǁpredict__mutmut_8': xǁHypeCycleModelǁpredict__mutmut_8, 
        'xǁHypeCycleModelǁpredict__mutmut_9': xǁHypeCycleModelǁpredict__mutmut_9, 
        'xǁHypeCycleModelǁpredict__mutmut_10': xǁHypeCycleModelǁpredict__mutmut_10, 
        'xǁHypeCycleModelǁpredict__mutmut_11': xǁHypeCycleModelǁpredict__mutmut_11, 
        'xǁHypeCycleModelǁpredict__mutmut_12': xǁHypeCycleModelǁpredict__mutmut_12, 
        'xǁHypeCycleModelǁpredict__mutmut_13': xǁHypeCycleModelǁpredict__mutmut_13, 
        'xǁHypeCycleModelǁpredict__mutmut_14': xǁHypeCycleModelǁpredict__mutmut_14, 
        'xǁHypeCycleModelǁpredict__mutmut_15': xǁHypeCycleModelǁpredict__mutmut_15, 
        'xǁHypeCycleModelǁpredict__mutmut_16': xǁHypeCycleModelǁpredict__mutmut_16, 
        'xǁHypeCycleModelǁpredict__mutmut_17': xǁHypeCycleModelǁpredict__mutmut_17, 
        'xǁHypeCycleModelǁpredict__mutmut_18': xǁHypeCycleModelǁpredict__mutmut_18, 
        'xǁHypeCycleModelǁpredict__mutmut_19': xǁHypeCycleModelǁpredict__mutmut_19, 
        'xǁHypeCycleModelǁpredict__mutmut_20': xǁHypeCycleModelǁpredict__mutmut_20, 
        'xǁHypeCycleModelǁpredict__mutmut_21': xǁHypeCycleModelǁpredict__mutmut_21, 
        'xǁHypeCycleModelǁpredict__mutmut_22': xǁHypeCycleModelǁpredict__mutmut_22, 
        'xǁHypeCycleModelǁpredict__mutmut_23': xǁHypeCycleModelǁpredict__mutmut_23, 
        'xǁHypeCycleModelǁpredict__mutmut_24': xǁHypeCycleModelǁpredict__mutmut_24, 
        'xǁHypeCycleModelǁpredict__mutmut_25': xǁHypeCycleModelǁpredict__mutmut_25, 
        'xǁHypeCycleModelǁpredict__mutmut_26': xǁHypeCycleModelǁpredict__mutmut_26, 
        'xǁHypeCycleModelǁpredict__mutmut_27': xǁHypeCycleModelǁpredict__mutmut_27, 
        'xǁHypeCycleModelǁpredict__mutmut_28': xǁHypeCycleModelǁpredict__mutmut_28, 
        'xǁHypeCycleModelǁpredict__mutmut_29': xǁHypeCycleModelǁpredict__mutmut_29, 
        'xǁHypeCycleModelǁpredict__mutmut_30': xǁHypeCycleModelǁpredict__mutmut_30, 
        'xǁHypeCycleModelǁpredict__mutmut_31': xǁHypeCycleModelǁpredict__mutmut_31, 
        'xǁHypeCycleModelǁpredict__mutmut_32': xǁHypeCycleModelǁpredict__mutmut_32, 
        'xǁHypeCycleModelǁpredict__mutmut_33': xǁHypeCycleModelǁpredict__mutmut_33, 
        'xǁHypeCycleModelǁpredict__mutmut_34': xǁHypeCycleModelǁpredict__mutmut_34, 
        'xǁHypeCycleModelǁpredict__mutmut_35': xǁHypeCycleModelǁpredict__mutmut_35, 
        'xǁHypeCycleModelǁpredict__mutmut_36': xǁHypeCycleModelǁpredict__mutmut_36, 
        'xǁHypeCycleModelǁpredict__mutmut_37': xǁHypeCycleModelǁpredict__mutmut_37, 
        'xǁHypeCycleModelǁpredict__mutmut_38': xǁHypeCycleModelǁpredict__mutmut_38, 
        'xǁHypeCycleModelǁpredict__mutmut_39': xǁHypeCycleModelǁpredict__mutmut_39, 
        'xǁHypeCycleModelǁpredict__mutmut_40': xǁHypeCycleModelǁpredict__mutmut_40, 
        'xǁHypeCycleModelǁpredict__mutmut_41': xǁHypeCycleModelǁpredict__mutmut_41, 
        'xǁHypeCycleModelǁpredict__mutmut_42': xǁHypeCycleModelǁpredict__mutmut_42, 
        'xǁHypeCycleModelǁpredict__mutmut_43': xǁHypeCycleModelǁpredict__mutmut_43, 
        'xǁHypeCycleModelǁpredict__mutmut_44': xǁHypeCycleModelǁpredict__mutmut_44, 
        'xǁHypeCycleModelǁpredict__mutmut_45': xǁHypeCycleModelǁpredict__mutmut_45, 
        'xǁHypeCycleModelǁpredict__mutmut_46': xǁHypeCycleModelǁpredict__mutmut_46, 
        'xǁHypeCycleModelǁpredict__mutmut_47': xǁHypeCycleModelǁpredict__mutmut_47, 
        'xǁHypeCycleModelǁpredict__mutmut_48': xǁHypeCycleModelǁpredict__mutmut_48, 
        'xǁHypeCycleModelǁpredict__mutmut_49': xǁHypeCycleModelǁpredict__mutmut_49, 
        'xǁHypeCycleModelǁpredict__mutmut_50': xǁHypeCycleModelǁpredict__mutmut_50, 
        'xǁHypeCycleModelǁpredict__mutmut_51': xǁHypeCycleModelǁpredict__mutmut_51, 
        'xǁHypeCycleModelǁpredict__mutmut_52': xǁHypeCycleModelǁpredict__mutmut_52, 
        'xǁHypeCycleModelǁpredict__mutmut_53': xǁHypeCycleModelǁpredict__mutmut_53, 
        'xǁHypeCycleModelǁpredict__mutmut_54': xǁHypeCycleModelǁpredict__mutmut_54, 
        'xǁHypeCycleModelǁpredict__mutmut_55': xǁHypeCycleModelǁpredict__mutmut_55, 
        'xǁHypeCycleModelǁpredict__mutmut_56': xǁHypeCycleModelǁpredict__mutmut_56, 
        'xǁHypeCycleModelǁpredict__mutmut_57': xǁHypeCycleModelǁpredict__mutmut_57, 
        'xǁHypeCycleModelǁpredict__mutmut_58': xǁHypeCycleModelǁpredict__mutmut_58, 
        'xǁHypeCycleModelǁpredict__mutmut_59': xǁHypeCycleModelǁpredict__mutmut_59, 
        'xǁHypeCycleModelǁpredict__mutmut_60': xǁHypeCycleModelǁpredict__mutmut_60, 
        'xǁHypeCycleModelǁpredict__mutmut_61': xǁHypeCycleModelǁpredict__mutmut_61, 
        'xǁHypeCycleModelǁpredict__mutmut_62': xǁHypeCycleModelǁpredict__mutmut_62, 
        'xǁHypeCycleModelǁpredict__mutmut_63': xǁHypeCycleModelǁpredict__mutmut_63, 
        'xǁHypeCycleModelǁpredict__mutmut_64': xǁHypeCycleModelǁpredict__mutmut_64, 
        'xǁHypeCycleModelǁpredict__mutmut_65': xǁHypeCycleModelǁpredict__mutmut_65, 
        'xǁHypeCycleModelǁpredict__mutmut_66': xǁHypeCycleModelǁpredict__mutmut_66, 
        'xǁHypeCycleModelǁpredict__mutmut_67': xǁHypeCycleModelǁpredict__mutmut_67, 
        'xǁHypeCycleModelǁpredict__mutmut_68': xǁHypeCycleModelǁpredict__mutmut_68, 
        'xǁHypeCycleModelǁpredict__mutmut_69': xǁHypeCycleModelǁpredict__mutmut_69, 
        'xǁHypeCycleModelǁpredict__mutmut_70': xǁHypeCycleModelǁpredict__mutmut_70, 
        'xǁHypeCycleModelǁpredict__mutmut_71': xǁHypeCycleModelǁpredict__mutmut_71, 
        'xǁHypeCycleModelǁpredict__mutmut_72': xǁHypeCycleModelǁpredict__mutmut_72, 
        'xǁHypeCycleModelǁpredict__mutmut_73': xǁHypeCycleModelǁpredict__mutmut_73, 
        'xǁHypeCycleModelǁpredict__mutmut_74': xǁHypeCycleModelǁpredict__mutmut_74
    }
    xǁHypeCycleModelǁpredict__mutmut_orig.__name__ = 'xǁHypeCycleModelǁpredict'

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = value
