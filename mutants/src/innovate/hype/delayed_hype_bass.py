# src/innovate/hype/delayed_hype_bass.py

from collections.abc import Sequence

import numpy as np
from jitcdde import jitcdde, y
from jitcdde import t as time
from symengine import exp

from innovate.diffuse.bass import BassModel

from .hype_cycle import HypeCycleModel
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


class DelayedHypeBassModel:
    """A modified Bass model with a time-delayed hype influence, implemented
    using Delay Differential Equations (DDEs).
    """

    def __init__(self, bass_model: BassModel, hype_model: HypeCycleModel, delay: float):
        args = [bass_model, hype_model, delay]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDelayedHypeBassModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁDelayedHypeBassModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁDelayedHypeBassModelǁ__init____mutmut_orig(self, bass_model: BassModel, hype_model: HypeCycleModel, delay: float):
        self.bass_model = bass_model
        self.hype_model = hype_model
        self.delay = delay

    def xǁDelayedHypeBassModelǁ__init____mutmut_1(self, bass_model: BassModel, hype_model: HypeCycleModel, delay: float):
        self.bass_model = None
        self.hype_model = hype_model
        self.delay = delay

    def xǁDelayedHypeBassModelǁ__init____mutmut_2(self, bass_model: BassModel, hype_model: HypeCycleModel, delay: float):
        self.bass_model = bass_model
        self.hype_model = None
        self.delay = delay

    def xǁDelayedHypeBassModelǁ__init____mutmut_3(self, bass_model: BassModel, hype_model: HypeCycleModel, delay: float):
        self.bass_model = bass_model
        self.hype_model = hype_model
        self.delay = None
    
    xǁDelayedHypeBassModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDelayedHypeBassModelǁ__init____mutmut_1': xǁDelayedHypeBassModelǁ__init____mutmut_1, 
        'xǁDelayedHypeBassModelǁ__init____mutmut_2': xǁDelayedHypeBassModelǁ__init____mutmut_2, 
        'xǁDelayedHypeBassModelǁ__init____mutmut_3': xǁDelayedHypeBassModelǁ__init____mutmut_3
    }
    xǁDelayedHypeBassModelǁ__init____mutmut_orig.__name__ = 'xǁDelayedHypeBassModelǁ__init__'

    def predict(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        args = [t_eval, y0]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDelayedHypeBassModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁDelayedHypeBassModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁDelayedHypeBassModelǁpredict__mutmut_orig(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_1(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ and not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_2(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_3(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_4(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                None,
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_5(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "XXBoth the Bass and Hype models must have parameters set.XX",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_6(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "both the bass and hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_7(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "BOTH THE BASS AND HYPE MODELS MUST HAVE PARAMETERS SET.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_8(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = None
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_9(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["XXpXX"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_10(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["P"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_11(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = None
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_12(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["XXqXX"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_13(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["Q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_14(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = None

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_15(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["XXmXX"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_16(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["M"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_17(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = None
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_18(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = None

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_19(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["XXkXX"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_20(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["K"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_21(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["XXt0XX"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_22(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["T0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_23(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["XXa_hypeXX"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_24(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["A_HYPE"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_25(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["XXt_hypeXX"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_26(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["T_HYPE"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_27(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["XXw_hypeXX"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_28(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["W_HYPE"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_29(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["XXa_dXX"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_30(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["A_D"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_31(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["XXt_dXX"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_32(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["T_D"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_33(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["XXw_dXX"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_34(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["W_D"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_35(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = None
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_36(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 * (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_37(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 1.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_38(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 - exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_39(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (2 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_40(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(None))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_41(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k / (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_42(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(+k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_43(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time + t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_44(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = None
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_45(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h / exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_46(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(None)
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_47(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) * (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_48(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(+((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_49(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) * 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_50(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time + t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_51(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 3) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_52(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 / w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_53(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (3 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_54(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h * 2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_55(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**3))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_56(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = None
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_57(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d / exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_58(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(None)
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_59(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) * (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_60(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(+((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_61(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) * 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_62(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time + t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_63(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 3) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_64(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 / w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_65(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (3 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_66(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d * 2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_67(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**3))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_68(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = None

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_69(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype + disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_70(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity - hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_71(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = None

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_72(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0)) - (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_73(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) / (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_74(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base / (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_75(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 - visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_76(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (2 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_77(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(None, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_78(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, None))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_79(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_80(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, ))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_81(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time + self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_82(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m + y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_83(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(None))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_84(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(1))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_85(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m / (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_86(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) * m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_87(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) / y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_88(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base / (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_89(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 - visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_90(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (2 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_91(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(None, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_92(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, None))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_93(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_94(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, ))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_95(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time + self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_96(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(None) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_97(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(1) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_98(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m + y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_99(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(None)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_100(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(1)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_101(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = None
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_102(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(None)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_103(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past(None)
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_104(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = None
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_105(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(None)

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_106(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(None)[0])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_107(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[1])

        return np.array(adoption)

    def xǁDelayedHypeBassModelǁpredict__mutmut_108(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time using a DDE solver."""
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay))) * y(0) / m * (m - y(0)),
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(None)
    
    xǁDelayedHypeBassModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDelayedHypeBassModelǁpredict__mutmut_1': xǁDelayedHypeBassModelǁpredict__mutmut_1, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_2': xǁDelayedHypeBassModelǁpredict__mutmut_2, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_3': xǁDelayedHypeBassModelǁpredict__mutmut_3, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_4': xǁDelayedHypeBassModelǁpredict__mutmut_4, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_5': xǁDelayedHypeBassModelǁpredict__mutmut_5, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_6': xǁDelayedHypeBassModelǁpredict__mutmut_6, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_7': xǁDelayedHypeBassModelǁpredict__mutmut_7, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_8': xǁDelayedHypeBassModelǁpredict__mutmut_8, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_9': xǁDelayedHypeBassModelǁpredict__mutmut_9, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_10': xǁDelayedHypeBassModelǁpredict__mutmut_10, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_11': xǁDelayedHypeBassModelǁpredict__mutmut_11, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_12': xǁDelayedHypeBassModelǁpredict__mutmut_12, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_13': xǁDelayedHypeBassModelǁpredict__mutmut_13, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_14': xǁDelayedHypeBassModelǁpredict__mutmut_14, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_15': xǁDelayedHypeBassModelǁpredict__mutmut_15, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_16': xǁDelayedHypeBassModelǁpredict__mutmut_16, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_17': xǁDelayedHypeBassModelǁpredict__mutmut_17, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_18': xǁDelayedHypeBassModelǁpredict__mutmut_18, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_19': xǁDelayedHypeBassModelǁpredict__mutmut_19, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_20': xǁDelayedHypeBassModelǁpredict__mutmut_20, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_21': xǁDelayedHypeBassModelǁpredict__mutmut_21, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_22': xǁDelayedHypeBassModelǁpredict__mutmut_22, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_23': xǁDelayedHypeBassModelǁpredict__mutmut_23, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_24': xǁDelayedHypeBassModelǁpredict__mutmut_24, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_25': xǁDelayedHypeBassModelǁpredict__mutmut_25, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_26': xǁDelayedHypeBassModelǁpredict__mutmut_26, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_27': xǁDelayedHypeBassModelǁpredict__mutmut_27, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_28': xǁDelayedHypeBassModelǁpredict__mutmut_28, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_29': xǁDelayedHypeBassModelǁpredict__mutmut_29, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_30': xǁDelayedHypeBassModelǁpredict__mutmut_30, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_31': xǁDelayedHypeBassModelǁpredict__mutmut_31, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_32': xǁDelayedHypeBassModelǁpredict__mutmut_32, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_33': xǁDelayedHypeBassModelǁpredict__mutmut_33, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_34': xǁDelayedHypeBassModelǁpredict__mutmut_34, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_35': xǁDelayedHypeBassModelǁpredict__mutmut_35, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_36': xǁDelayedHypeBassModelǁpredict__mutmut_36, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_37': xǁDelayedHypeBassModelǁpredict__mutmut_37, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_38': xǁDelayedHypeBassModelǁpredict__mutmut_38, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_39': xǁDelayedHypeBassModelǁpredict__mutmut_39, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_40': xǁDelayedHypeBassModelǁpredict__mutmut_40, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_41': xǁDelayedHypeBassModelǁpredict__mutmut_41, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_42': xǁDelayedHypeBassModelǁpredict__mutmut_42, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_43': xǁDelayedHypeBassModelǁpredict__mutmut_43, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_44': xǁDelayedHypeBassModelǁpredict__mutmut_44, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_45': xǁDelayedHypeBassModelǁpredict__mutmut_45, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_46': xǁDelayedHypeBassModelǁpredict__mutmut_46, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_47': xǁDelayedHypeBassModelǁpredict__mutmut_47, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_48': xǁDelayedHypeBassModelǁpredict__mutmut_48, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_49': xǁDelayedHypeBassModelǁpredict__mutmut_49, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_50': xǁDelayedHypeBassModelǁpredict__mutmut_50, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_51': xǁDelayedHypeBassModelǁpredict__mutmut_51, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_52': xǁDelayedHypeBassModelǁpredict__mutmut_52, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_53': xǁDelayedHypeBassModelǁpredict__mutmut_53, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_54': xǁDelayedHypeBassModelǁpredict__mutmut_54, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_55': xǁDelayedHypeBassModelǁpredict__mutmut_55, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_56': xǁDelayedHypeBassModelǁpredict__mutmut_56, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_57': xǁDelayedHypeBassModelǁpredict__mutmut_57, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_58': xǁDelayedHypeBassModelǁpredict__mutmut_58, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_59': xǁDelayedHypeBassModelǁpredict__mutmut_59, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_60': xǁDelayedHypeBassModelǁpredict__mutmut_60, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_61': xǁDelayedHypeBassModelǁpredict__mutmut_61, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_62': xǁDelayedHypeBassModelǁpredict__mutmut_62, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_63': xǁDelayedHypeBassModelǁpredict__mutmut_63, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_64': xǁDelayedHypeBassModelǁpredict__mutmut_64, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_65': xǁDelayedHypeBassModelǁpredict__mutmut_65, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_66': xǁDelayedHypeBassModelǁpredict__mutmut_66, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_67': xǁDelayedHypeBassModelǁpredict__mutmut_67, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_68': xǁDelayedHypeBassModelǁpredict__mutmut_68, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_69': xǁDelayedHypeBassModelǁpredict__mutmut_69, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_70': xǁDelayedHypeBassModelǁpredict__mutmut_70, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_71': xǁDelayedHypeBassModelǁpredict__mutmut_71, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_72': xǁDelayedHypeBassModelǁpredict__mutmut_72, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_73': xǁDelayedHypeBassModelǁpredict__mutmut_73, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_74': xǁDelayedHypeBassModelǁpredict__mutmut_74, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_75': xǁDelayedHypeBassModelǁpredict__mutmut_75, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_76': xǁDelayedHypeBassModelǁpredict__mutmut_76, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_77': xǁDelayedHypeBassModelǁpredict__mutmut_77, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_78': xǁDelayedHypeBassModelǁpredict__mutmut_78, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_79': xǁDelayedHypeBassModelǁpredict__mutmut_79, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_80': xǁDelayedHypeBassModelǁpredict__mutmut_80, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_81': xǁDelayedHypeBassModelǁpredict__mutmut_81, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_82': xǁDelayedHypeBassModelǁpredict__mutmut_82, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_83': xǁDelayedHypeBassModelǁpredict__mutmut_83, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_84': xǁDelayedHypeBassModelǁpredict__mutmut_84, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_85': xǁDelayedHypeBassModelǁpredict__mutmut_85, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_86': xǁDelayedHypeBassModelǁpredict__mutmut_86, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_87': xǁDelayedHypeBassModelǁpredict__mutmut_87, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_88': xǁDelayedHypeBassModelǁpredict__mutmut_88, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_89': xǁDelayedHypeBassModelǁpredict__mutmut_89, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_90': xǁDelayedHypeBassModelǁpredict__mutmut_90, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_91': xǁDelayedHypeBassModelǁpredict__mutmut_91, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_92': xǁDelayedHypeBassModelǁpredict__mutmut_92, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_93': xǁDelayedHypeBassModelǁpredict__mutmut_93, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_94': xǁDelayedHypeBassModelǁpredict__mutmut_94, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_95': xǁDelayedHypeBassModelǁpredict__mutmut_95, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_96': xǁDelayedHypeBassModelǁpredict__mutmut_96, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_97': xǁDelayedHypeBassModelǁpredict__mutmut_97, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_98': xǁDelayedHypeBassModelǁpredict__mutmut_98, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_99': xǁDelayedHypeBassModelǁpredict__mutmut_99, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_100': xǁDelayedHypeBassModelǁpredict__mutmut_100, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_101': xǁDelayedHypeBassModelǁpredict__mutmut_101, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_102': xǁDelayedHypeBassModelǁpredict__mutmut_102, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_103': xǁDelayedHypeBassModelǁpredict__mutmut_103, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_104': xǁDelayedHypeBassModelǁpredict__mutmut_104, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_105': xǁDelayedHypeBassModelǁpredict__mutmut_105, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_106': xǁDelayedHypeBassModelǁpredict__mutmut_106, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_107': xǁDelayedHypeBassModelǁpredict__mutmut_107, 
        'xǁDelayedHypeBassModelǁpredict__mutmut_108': xǁDelayedHypeBassModelǁpredict__mutmut_108
    }
    xǁDelayedHypeBassModelǁpredict__mutmut_orig.__name__ = 'xǁDelayedHypeBassModelǁpredict'
