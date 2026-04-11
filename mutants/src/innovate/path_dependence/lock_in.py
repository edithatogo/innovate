from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.integrate import odeint
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


class LockInModel:
    """A simple model demonstrating path dependence and lock-in effects
    between two competing technologies.

    The model simulates two technologies where the growth rate of each
    is positively influenced by its own installed base (network effects)
    and negatively by the competitor's.
    """

    def __init__(self) -> None:
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLockInModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁLockInModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁLockInModelǁ__init____mutmut_orig(self) -> None:
        self._params: dict[str, float] = {}

    def xǁLockInModelǁ__init____mutmut_1(self) -> None:
        self._params: dict[str, float] = None
    
    xǁLockInModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLockInModelǁ__init____mutmut_1': xǁLockInModelǁ__init____mutmut_1
    }
    xǁLockInModelǁ__init____mutmut_orig.__name__ = 'xǁLockInModelǁ__init__'

    @property
    def param_names(self) -> Sequence[str]:
        return [
            "alpha1",  # Intrinsic growth rate of Tech 1
            "alpha2",  # Intrinsic growth rate of Tech 2
            "beta1",  # Network effect strength for Tech 1
            "beta2",  # Network effect strength for Tech 2
            "gamma1",  # Negative influence of Tech 2 on Tech 1
            "gamma2",  # Negative influence of Tech 1 on Tech 2
            "m",  # Total market potential (assumed shared)
        ]

    def initial_guesses(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLockInModelǁinitial_guesses__mutmut_orig'), object.__getattribute__(self, 'xǁLockInModelǁinitial_guesses__mutmut_mutants'), args, kwargs, self)

    def xǁLockInModelǁinitial_guesses__mutmut_orig(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_1(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = None
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_2(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(None)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_3(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "XXalpha1XX": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_4(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "ALPHA1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_5(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 1.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_6(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "XXalpha2XX": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_7(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "ALPHA2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_8(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 1.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_9(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "XXbeta1XX": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_10(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "BETA1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_11(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 1.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_12(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "XXbeta2XX": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_13(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "BETA2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_14(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 1.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_15(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "XXgamma1XX": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_16(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "GAMMA1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_17(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 1.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_18(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "XXgamma2XX": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_19(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "GAMMA2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_20(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 1.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_21(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "XXmXX": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_22(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "M": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_23(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y / 1.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_24(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 2.5 if max_y > 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_25(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y >= 0 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_26(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 1 else 1000.0,
        }

    def xǁLockInModelǁinitial_guesses__mutmut_27(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1001.0,
        }
    
    xǁLockInModelǁinitial_guesses__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLockInModelǁinitial_guesses__mutmut_1': xǁLockInModelǁinitial_guesses__mutmut_1, 
        'xǁLockInModelǁinitial_guesses__mutmut_2': xǁLockInModelǁinitial_guesses__mutmut_2, 
        'xǁLockInModelǁinitial_guesses__mutmut_3': xǁLockInModelǁinitial_guesses__mutmut_3, 
        'xǁLockInModelǁinitial_guesses__mutmut_4': xǁLockInModelǁinitial_guesses__mutmut_4, 
        'xǁLockInModelǁinitial_guesses__mutmut_5': xǁLockInModelǁinitial_guesses__mutmut_5, 
        'xǁLockInModelǁinitial_guesses__mutmut_6': xǁLockInModelǁinitial_guesses__mutmut_6, 
        'xǁLockInModelǁinitial_guesses__mutmut_7': xǁLockInModelǁinitial_guesses__mutmut_7, 
        'xǁLockInModelǁinitial_guesses__mutmut_8': xǁLockInModelǁinitial_guesses__mutmut_8, 
        'xǁLockInModelǁinitial_guesses__mutmut_9': xǁLockInModelǁinitial_guesses__mutmut_9, 
        'xǁLockInModelǁinitial_guesses__mutmut_10': xǁLockInModelǁinitial_guesses__mutmut_10, 
        'xǁLockInModelǁinitial_guesses__mutmut_11': xǁLockInModelǁinitial_guesses__mutmut_11, 
        'xǁLockInModelǁinitial_guesses__mutmut_12': xǁLockInModelǁinitial_guesses__mutmut_12, 
        'xǁLockInModelǁinitial_guesses__mutmut_13': xǁLockInModelǁinitial_guesses__mutmut_13, 
        'xǁLockInModelǁinitial_guesses__mutmut_14': xǁLockInModelǁinitial_guesses__mutmut_14, 
        'xǁLockInModelǁinitial_guesses__mutmut_15': xǁLockInModelǁinitial_guesses__mutmut_15, 
        'xǁLockInModelǁinitial_guesses__mutmut_16': xǁLockInModelǁinitial_guesses__mutmut_16, 
        'xǁLockInModelǁinitial_guesses__mutmut_17': xǁLockInModelǁinitial_guesses__mutmut_17, 
        'xǁLockInModelǁinitial_guesses__mutmut_18': xǁLockInModelǁinitial_guesses__mutmut_18, 
        'xǁLockInModelǁinitial_guesses__mutmut_19': xǁLockInModelǁinitial_guesses__mutmut_19, 
        'xǁLockInModelǁinitial_guesses__mutmut_20': xǁLockInModelǁinitial_guesses__mutmut_20, 
        'xǁLockInModelǁinitial_guesses__mutmut_21': xǁLockInModelǁinitial_guesses__mutmut_21, 
        'xǁLockInModelǁinitial_guesses__mutmut_22': xǁLockInModelǁinitial_guesses__mutmut_22, 
        'xǁLockInModelǁinitial_guesses__mutmut_23': xǁLockInModelǁinitial_guesses__mutmut_23, 
        'xǁLockInModelǁinitial_guesses__mutmut_24': xǁLockInModelǁinitial_guesses__mutmut_24, 
        'xǁLockInModelǁinitial_guesses__mutmut_25': xǁLockInModelǁinitial_guesses__mutmut_25, 
        'xǁLockInModelǁinitial_guesses__mutmut_26': xǁLockInModelǁinitial_guesses__mutmut_26, 
        'xǁLockInModelǁinitial_guesses__mutmut_27': xǁLockInModelǁinitial_guesses__mutmut_27
    }
    xǁLockInModelǁinitial_guesses__mutmut_orig.__name__ = 'xǁLockInModelǁinitial_guesses'

    def bounds(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLockInModelǁbounds__mutmut_orig'), object.__getattribute__(self, 'xǁLockInModelǁbounds__mutmut_mutants'), args, kwargs, self)

    def xǁLockInModelǁbounds__mutmut_orig(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_1(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = None
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_2(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(None)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_3(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "XXalpha1XX": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_4(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "ALPHA1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_5(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (1, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_6(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "XXalpha2XX": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_7(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "ALPHA2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_8(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (1, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_9(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "XXbeta1XX": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_10(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "BETA1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_11(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (1, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_12(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "XXbeta2XX": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_13(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "BETA2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_14(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (1, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_15(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "XXgamma1XX": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_16(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "GAMMA1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_17(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (1, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_18(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "XXgamma2XX": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_19(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "GAMMA2": (0, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_20(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (1, np.inf),
            "m": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_21(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "XXmXX": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_22(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "M": (float(max_y), np.inf),
        }

    def xǁLockInModelǁbounds__mutmut_23(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (float(None), np.inf),
        }
    
    xǁLockInModelǁbounds__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLockInModelǁbounds__mutmut_1': xǁLockInModelǁbounds__mutmut_1, 
        'xǁLockInModelǁbounds__mutmut_2': xǁLockInModelǁbounds__mutmut_2, 
        'xǁLockInModelǁbounds__mutmut_3': xǁLockInModelǁbounds__mutmut_3, 
        'xǁLockInModelǁbounds__mutmut_4': xǁLockInModelǁbounds__mutmut_4, 
        'xǁLockInModelǁbounds__mutmut_5': xǁLockInModelǁbounds__mutmut_5, 
        'xǁLockInModelǁbounds__mutmut_6': xǁLockInModelǁbounds__mutmut_6, 
        'xǁLockInModelǁbounds__mutmut_7': xǁLockInModelǁbounds__mutmut_7, 
        'xǁLockInModelǁbounds__mutmut_8': xǁLockInModelǁbounds__mutmut_8, 
        'xǁLockInModelǁbounds__mutmut_9': xǁLockInModelǁbounds__mutmut_9, 
        'xǁLockInModelǁbounds__mutmut_10': xǁLockInModelǁbounds__mutmut_10, 
        'xǁLockInModelǁbounds__mutmut_11': xǁLockInModelǁbounds__mutmut_11, 
        'xǁLockInModelǁbounds__mutmut_12': xǁLockInModelǁbounds__mutmut_12, 
        'xǁLockInModelǁbounds__mutmut_13': xǁLockInModelǁbounds__mutmut_13, 
        'xǁLockInModelǁbounds__mutmut_14': xǁLockInModelǁbounds__mutmut_14, 
        'xǁLockInModelǁbounds__mutmut_15': xǁLockInModelǁbounds__mutmut_15, 
        'xǁLockInModelǁbounds__mutmut_16': xǁLockInModelǁbounds__mutmut_16, 
        'xǁLockInModelǁbounds__mutmut_17': xǁLockInModelǁbounds__mutmut_17, 
        'xǁLockInModelǁbounds__mutmut_18': xǁLockInModelǁbounds__mutmut_18, 
        'xǁLockInModelǁbounds__mutmut_19': xǁLockInModelǁbounds__mutmut_19, 
        'xǁLockInModelǁbounds__mutmut_20': xǁLockInModelǁbounds__mutmut_20, 
        'xǁLockInModelǁbounds__mutmut_21': xǁLockInModelǁbounds__mutmut_21, 
        'xǁLockInModelǁbounds__mutmut_22': xǁLockInModelǁbounds__mutmut_22, 
        'xǁLockInModelǁbounds__mutmut_23': xǁLockInModelǁbounds__mutmut_23
    }
    xǁLockInModelǁbounds__mutmut_orig.__name__ = 'xǁLockInModelǁbounds'

    @staticmethod
    def differential_equation(y_current: np.ndarray, t_current: float, *params: float) -> Sequence[float]:
        alpha1, alpha2, beta1, beta2, gamma1, gamma2, m = params
        n1, n2 = y_current

        # Ensure populations are non-negative and do not exceed market potential
        n1 = max(0, min(n1, m))
        n2 = max(0, min(n2, m))

        # Simple logistic-like growth with network effects and competition
        dn1_dt = alpha1 * n1 * (1 - (n1 + n2) / m) + beta1 * n1 * (n1 / m) - gamma1 * n1 * (n2 / m)
        dn2_dt = alpha2 * n2 * (1 - (n1 + n2) / m) + beta2 * n2 * (n2 / m) - gamma2 * n2 * (n1 / m)

        return [dn1_dt, dn2_dt]

    def predict(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        args = [t, y0]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLockInModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁLockInModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁLockInModelǁpredict__mutmut_orig(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_1(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_2(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError(None)

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_3(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_4(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_5(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_6(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = None
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_7(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            None,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_8(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            None,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_9(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            None,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_10(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=None,
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_11(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_12(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_13(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_14(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_15(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(None),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_16(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = None
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_17(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(None, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_18(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, None)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_19(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_20(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, )
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_21(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(1, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_22(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = None
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_23(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get(None, np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_24(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", None)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_25(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get(np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_26(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", )
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_27(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("XXmXX", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_28(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("M", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_29(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = None
        return sol

    def xǁLockInModelǁpredict__mutmut_30(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(None, m)
        return sol

    def xǁLockInModelǁpredict__mutmut_31(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, None)
        return sol

    def xǁLockInModelǁpredict__mutmut_32(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(m)
        return sol

    def xǁLockInModelǁpredict__mutmut_33(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, )
        return sol
    
    xǁLockInModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLockInModelǁpredict__mutmut_1': xǁLockInModelǁpredict__mutmut_1, 
        'xǁLockInModelǁpredict__mutmut_2': xǁLockInModelǁpredict__mutmut_2, 
        'xǁLockInModelǁpredict__mutmut_3': xǁLockInModelǁpredict__mutmut_3, 
        'xǁLockInModelǁpredict__mutmut_4': xǁLockInModelǁpredict__mutmut_4, 
        'xǁLockInModelǁpredict__mutmut_5': xǁLockInModelǁpredict__mutmut_5, 
        'xǁLockInModelǁpredict__mutmut_6': xǁLockInModelǁpredict__mutmut_6, 
        'xǁLockInModelǁpredict__mutmut_7': xǁLockInModelǁpredict__mutmut_7, 
        'xǁLockInModelǁpredict__mutmut_8': xǁLockInModelǁpredict__mutmut_8, 
        'xǁLockInModelǁpredict__mutmut_9': xǁLockInModelǁpredict__mutmut_9, 
        'xǁLockInModelǁpredict__mutmut_10': xǁLockInModelǁpredict__mutmut_10, 
        'xǁLockInModelǁpredict__mutmut_11': xǁLockInModelǁpredict__mutmut_11, 
        'xǁLockInModelǁpredict__mutmut_12': xǁLockInModelǁpredict__mutmut_12, 
        'xǁLockInModelǁpredict__mutmut_13': xǁLockInModelǁpredict__mutmut_13, 
        'xǁLockInModelǁpredict__mutmut_14': xǁLockInModelǁpredict__mutmut_14, 
        'xǁLockInModelǁpredict__mutmut_15': xǁLockInModelǁpredict__mutmut_15, 
        'xǁLockInModelǁpredict__mutmut_16': xǁLockInModelǁpredict__mutmut_16, 
        'xǁLockInModelǁpredict__mutmut_17': xǁLockInModelǁpredict__mutmut_17, 
        'xǁLockInModelǁpredict__mutmut_18': xǁLockInModelǁpredict__mutmut_18, 
        'xǁLockInModelǁpredict__mutmut_19': xǁLockInModelǁpredict__mutmut_19, 
        'xǁLockInModelǁpredict__mutmut_20': xǁLockInModelǁpredict__mutmut_20, 
        'xǁLockInModelǁpredict__mutmut_21': xǁLockInModelǁpredict__mutmut_21, 
        'xǁLockInModelǁpredict__mutmut_22': xǁLockInModelǁpredict__mutmut_22, 
        'xǁLockInModelǁpredict__mutmut_23': xǁLockInModelǁpredict__mutmut_23, 
        'xǁLockInModelǁpredict__mutmut_24': xǁLockInModelǁpredict__mutmut_24, 
        'xǁLockInModelǁpredict__mutmut_25': xǁLockInModelǁpredict__mutmut_25, 
        'xǁLockInModelǁpredict__mutmut_26': xǁLockInModelǁpredict__mutmut_26, 
        'xǁLockInModelǁpredict__mutmut_27': xǁLockInModelǁpredict__mutmut_27, 
        'xǁLockInModelǁpredict__mutmut_28': xǁLockInModelǁpredict__mutmut_28, 
        'xǁLockInModelǁpredict__mutmut_29': xǁLockInModelǁpredict__mutmut_29, 
        'xǁLockInModelǁpredict__mutmut_30': xǁLockInModelǁpredict__mutmut_30, 
        'xǁLockInModelǁpredict__mutmut_31': xǁLockInModelǁpredict__mutmut_31, 
        'xǁLockInModelǁpredict__mutmut_32': xǁLockInModelǁpredict__mutmut_32, 
        'xǁLockInModelǁpredict__mutmut_33': xǁLockInModelǁpredict__mutmut_33
    }
    xǁLockInModelǁpredict__mutmut_orig.__name__ = 'xǁLockInModelǁpredict'

    def fit(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        args = [t, y]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLockInModelǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁLockInModelǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁLockInModelǁfit__mutmut_orig(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_1(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = None
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_2(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(None)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_3(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 and y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_4(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim == 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_5(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 3 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_6(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[2] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_7(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] == 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_8(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 3:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_9(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                None,
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_10(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "XX`y` must be a 2D array with two columns (for two technologies).XX",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_11(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2d array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_12(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`Y` MUST BE A 2D ARRAY WITH TWO COLUMNS (FOR TWO TECHNOLOGIES).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_13(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = None

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_14(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[1, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_15(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = None
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_16(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(None)
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_17(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(None, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_18(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, None))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_19(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_20(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, ))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_21(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = None
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_22(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(None, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_23(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, None)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_24(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_25(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, )
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_26(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(None)

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_27(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum(None))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_28(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) * 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_29(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs + y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_30(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 3))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_31(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = None
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_32(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(None)
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_33(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(None, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_34(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, None).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_35(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_36(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, ).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_37(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = None

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_38(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(None)

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_39(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(None, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_40(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, None).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_41(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_42(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, ).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_43(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = None

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_44(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            None,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_45(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            None,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_46(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=None,
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_47(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=None,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_48(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method=None,
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_49(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_50(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_51(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_52(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_53(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_54(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_55(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="XXL-BFGS-BXX",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_56(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="l-bfgs-b",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_57(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_58(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(None)

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_59(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = None
        return self

    def xǁLockInModelǁfit__mutmut_60(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(None)
        return self

    def xǁLockInModelǁfit__mutmut_61(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(None, result.x))
        return self

    def xǁLockInModelǁfit__mutmut_62(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, None))
        return self

    def xǁLockInModelǁfit__mutmut_63(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(result.x))
        return self

    def xǁLockInModelǁfit__mutmut_64(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "LockInModel":
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "`y` must be a 2D array with two columns (for two technologies).",
            )

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y_obs: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y_obs - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method="L-BFGS-B",
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, ))
        return self
    
    xǁLockInModelǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLockInModelǁfit__mutmut_1': xǁLockInModelǁfit__mutmut_1, 
        'xǁLockInModelǁfit__mutmut_2': xǁLockInModelǁfit__mutmut_2, 
        'xǁLockInModelǁfit__mutmut_3': xǁLockInModelǁfit__mutmut_3, 
        'xǁLockInModelǁfit__mutmut_4': xǁLockInModelǁfit__mutmut_4, 
        'xǁLockInModelǁfit__mutmut_5': xǁLockInModelǁfit__mutmut_5, 
        'xǁLockInModelǁfit__mutmut_6': xǁLockInModelǁfit__mutmut_6, 
        'xǁLockInModelǁfit__mutmut_7': xǁLockInModelǁfit__mutmut_7, 
        'xǁLockInModelǁfit__mutmut_8': xǁLockInModelǁfit__mutmut_8, 
        'xǁLockInModelǁfit__mutmut_9': xǁLockInModelǁfit__mutmut_9, 
        'xǁLockInModelǁfit__mutmut_10': xǁLockInModelǁfit__mutmut_10, 
        'xǁLockInModelǁfit__mutmut_11': xǁLockInModelǁfit__mutmut_11, 
        'xǁLockInModelǁfit__mutmut_12': xǁLockInModelǁfit__mutmut_12, 
        'xǁLockInModelǁfit__mutmut_13': xǁLockInModelǁfit__mutmut_13, 
        'xǁLockInModelǁfit__mutmut_14': xǁLockInModelǁfit__mutmut_14, 
        'xǁLockInModelǁfit__mutmut_15': xǁLockInModelǁfit__mutmut_15, 
        'xǁLockInModelǁfit__mutmut_16': xǁLockInModelǁfit__mutmut_16, 
        'xǁLockInModelǁfit__mutmut_17': xǁLockInModelǁfit__mutmut_17, 
        'xǁLockInModelǁfit__mutmut_18': xǁLockInModelǁfit__mutmut_18, 
        'xǁLockInModelǁfit__mutmut_19': xǁLockInModelǁfit__mutmut_19, 
        'xǁLockInModelǁfit__mutmut_20': xǁLockInModelǁfit__mutmut_20, 
        'xǁLockInModelǁfit__mutmut_21': xǁLockInModelǁfit__mutmut_21, 
        'xǁLockInModelǁfit__mutmut_22': xǁLockInModelǁfit__mutmut_22, 
        'xǁLockInModelǁfit__mutmut_23': xǁLockInModelǁfit__mutmut_23, 
        'xǁLockInModelǁfit__mutmut_24': xǁLockInModelǁfit__mutmut_24, 
        'xǁLockInModelǁfit__mutmut_25': xǁLockInModelǁfit__mutmut_25, 
        'xǁLockInModelǁfit__mutmut_26': xǁLockInModelǁfit__mutmut_26, 
        'xǁLockInModelǁfit__mutmut_27': xǁLockInModelǁfit__mutmut_27, 
        'xǁLockInModelǁfit__mutmut_28': xǁLockInModelǁfit__mutmut_28, 
        'xǁLockInModelǁfit__mutmut_29': xǁLockInModelǁfit__mutmut_29, 
        'xǁLockInModelǁfit__mutmut_30': xǁLockInModelǁfit__mutmut_30, 
        'xǁLockInModelǁfit__mutmut_31': xǁLockInModelǁfit__mutmut_31, 
        'xǁLockInModelǁfit__mutmut_32': xǁLockInModelǁfit__mutmut_32, 
        'xǁLockInModelǁfit__mutmut_33': xǁLockInModelǁfit__mutmut_33, 
        'xǁLockInModelǁfit__mutmut_34': xǁLockInModelǁfit__mutmut_34, 
        'xǁLockInModelǁfit__mutmut_35': xǁLockInModelǁfit__mutmut_35, 
        'xǁLockInModelǁfit__mutmut_36': xǁLockInModelǁfit__mutmut_36, 
        'xǁLockInModelǁfit__mutmut_37': xǁLockInModelǁfit__mutmut_37, 
        'xǁLockInModelǁfit__mutmut_38': xǁLockInModelǁfit__mutmut_38, 
        'xǁLockInModelǁfit__mutmut_39': xǁLockInModelǁfit__mutmut_39, 
        'xǁLockInModelǁfit__mutmut_40': xǁLockInModelǁfit__mutmut_40, 
        'xǁLockInModelǁfit__mutmut_41': xǁLockInModelǁfit__mutmut_41, 
        'xǁLockInModelǁfit__mutmut_42': xǁLockInModelǁfit__mutmut_42, 
        'xǁLockInModelǁfit__mutmut_43': xǁLockInModelǁfit__mutmut_43, 
        'xǁLockInModelǁfit__mutmut_44': xǁLockInModelǁfit__mutmut_44, 
        'xǁLockInModelǁfit__mutmut_45': xǁLockInModelǁfit__mutmut_45, 
        'xǁLockInModelǁfit__mutmut_46': xǁLockInModelǁfit__mutmut_46, 
        'xǁLockInModelǁfit__mutmut_47': xǁLockInModelǁfit__mutmut_47, 
        'xǁLockInModelǁfit__mutmut_48': xǁLockInModelǁfit__mutmut_48, 
        'xǁLockInModelǁfit__mutmut_49': xǁLockInModelǁfit__mutmut_49, 
        'xǁLockInModelǁfit__mutmut_50': xǁLockInModelǁfit__mutmut_50, 
        'xǁLockInModelǁfit__mutmut_51': xǁLockInModelǁfit__mutmut_51, 
        'xǁLockInModelǁfit__mutmut_52': xǁLockInModelǁfit__mutmut_52, 
        'xǁLockInModelǁfit__mutmut_53': xǁLockInModelǁfit__mutmut_53, 
        'xǁLockInModelǁfit__mutmut_54': xǁLockInModelǁfit__mutmut_54, 
        'xǁLockInModelǁfit__mutmut_55': xǁLockInModelǁfit__mutmut_55, 
        'xǁLockInModelǁfit__mutmut_56': xǁLockInModelǁfit__mutmut_56, 
        'xǁLockInModelǁfit__mutmut_57': xǁLockInModelǁfit__mutmut_57, 
        'xǁLockInModelǁfit__mutmut_58': xǁLockInModelǁfit__mutmut_58, 
        'xǁLockInModelǁfit__mutmut_59': xǁLockInModelǁfit__mutmut_59, 
        'xǁLockInModelǁfit__mutmut_60': xǁLockInModelǁfit__mutmut_60, 
        'xǁLockInModelǁfit__mutmut_61': xǁLockInModelǁfit__mutmut_61, 
        'xǁLockInModelǁfit__mutmut_62': xǁLockInModelǁfit__mutmut_62, 
        'xǁLockInModelǁfit__mutmut_63': xǁLockInModelǁfit__mutmut_63, 
        'xǁLockInModelǁfit__mutmut_64': xǁLockInModelǁfit__mutmut_64
    }
    xǁLockInModelǁfit__mutmut_orig.__name__ = 'xǁLockInModelǁfit'

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]) -> None:
        self._params = value

    def score(self, t: Sequence[float], y: np.ndarray) -> float:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLockInModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁLockInModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁLockInModelǁscore__mutmut_orig(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_1(self, t: Sequence[float], y: np.ndarray) -> float:
        if self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_2(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError(None)
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_3(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet.XX")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_4(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_5(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_6(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = None
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_7(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(None, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_8(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, None)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_9(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_10(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, )
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_11(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[1, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_12(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = None
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_13(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum(None)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_14(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) * 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_15(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y + y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_16(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 3)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_17(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = None
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_18(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum(None)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_19(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) * 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_20(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y + np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_21(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(None, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_22(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=None)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_23(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_24(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, )) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_25(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=1)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_26(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 3)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_27(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_28(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_29(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0

    def xǁLockInModelǁscore__mutmut_30(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0

    def xǁLockInModelǁscore__mutmut_31(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0

    def xǁLockInModelǁscore__mutmut_32(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    xǁLockInModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLockInModelǁscore__mutmut_1': xǁLockInModelǁscore__mutmut_1, 
        'xǁLockInModelǁscore__mutmut_2': xǁLockInModelǁscore__mutmut_2, 
        'xǁLockInModelǁscore__mutmut_3': xǁLockInModelǁscore__mutmut_3, 
        'xǁLockInModelǁscore__mutmut_4': xǁLockInModelǁscore__mutmut_4, 
        'xǁLockInModelǁscore__mutmut_5': xǁLockInModelǁscore__mutmut_5, 
        'xǁLockInModelǁscore__mutmut_6': xǁLockInModelǁscore__mutmut_6, 
        'xǁLockInModelǁscore__mutmut_7': xǁLockInModelǁscore__mutmut_7, 
        'xǁLockInModelǁscore__mutmut_8': xǁLockInModelǁscore__mutmut_8, 
        'xǁLockInModelǁscore__mutmut_9': xǁLockInModelǁscore__mutmut_9, 
        'xǁLockInModelǁscore__mutmut_10': xǁLockInModelǁscore__mutmut_10, 
        'xǁLockInModelǁscore__mutmut_11': xǁLockInModelǁscore__mutmut_11, 
        'xǁLockInModelǁscore__mutmut_12': xǁLockInModelǁscore__mutmut_12, 
        'xǁLockInModelǁscore__mutmut_13': xǁLockInModelǁscore__mutmut_13, 
        'xǁLockInModelǁscore__mutmut_14': xǁLockInModelǁscore__mutmut_14, 
        'xǁLockInModelǁscore__mutmut_15': xǁLockInModelǁscore__mutmut_15, 
        'xǁLockInModelǁscore__mutmut_16': xǁLockInModelǁscore__mutmut_16, 
        'xǁLockInModelǁscore__mutmut_17': xǁLockInModelǁscore__mutmut_17, 
        'xǁLockInModelǁscore__mutmut_18': xǁLockInModelǁscore__mutmut_18, 
        'xǁLockInModelǁscore__mutmut_19': xǁLockInModelǁscore__mutmut_19, 
        'xǁLockInModelǁscore__mutmut_20': xǁLockInModelǁscore__mutmut_20, 
        'xǁLockInModelǁscore__mutmut_21': xǁLockInModelǁscore__mutmut_21, 
        'xǁLockInModelǁscore__mutmut_22': xǁLockInModelǁscore__mutmut_22, 
        'xǁLockInModelǁscore__mutmut_23': xǁLockInModelǁscore__mutmut_23, 
        'xǁLockInModelǁscore__mutmut_24': xǁLockInModelǁscore__mutmut_24, 
        'xǁLockInModelǁscore__mutmut_25': xǁLockInModelǁscore__mutmut_25, 
        'xǁLockInModelǁscore__mutmut_26': xǁLockInModelǁscore__mutmut_26, 
        'xǁLockInModelǁscore__mutmut_27': xǁLockInModelǁscore__mutmut_27, 
        'xǁLockInModelǁscore__mutmut_28': xǁLockInModelǁscore__mutmut_28, 
        'xǁLockInModelǁscore__mutmut_29': xǁLockInModelǁscore__mutmut_29, 
        'xǁLockInModelǁscore__mutmut_30': xǁLockInModelǁscore__mutmut_30, 
        'xǁLockInModelǁscore__mutmut_31': xǁLockInModelǁscore__mutmut_31, 
        'xǁLockInModelǁscore__mutmut_32': xǁLockInModelǁscore__mutmut_32
    }
    xǁLockInModelǁscore__mutmut_orig.__name__ = 'xǁLockInModelǁscore'

    def predict_adoption_rate(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        args = [t, y0]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLockInModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁLockInModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁLockInModelǁpredict_adoption_rate__mutmut_orig(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_1(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_2(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError(None)

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_3(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet.XX")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_4(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_5(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_6(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = None
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_7(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(None, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_8(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, None)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_9(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_10(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, )
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_11(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = None
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_12(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(None, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_13(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=None)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_14(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_15(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, )
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_16(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=1)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_17(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = None
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_18(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(None, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_19(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, None, *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_20(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(t[0], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_21(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_22(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], )
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_23(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[1], *self._params.values())
        return np.vstack([initial_rates, rates])

    def xǁLockInModelǁpredict_adoption_rate__mutmut_24(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        cumulative_predictions = self.predict(t, y0)
        rates = np.diff(cumulative_predictions, axis=0)
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack(None)
    
    xǁLockInModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLockInModelǁpredict_adoption_rate__mutmut_1': xǁLockInModelǁpredict_adoption_rate__mutmut_1, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_2': xǁLockInModelǁpredict_adoption_rate__mutmut_2, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_3': xǁLockInModelǁpredict_adoption_rate__mutmut_3, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_4': xǁLockInModelǁpredict_adoption_rate__mutmut_4, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_5': xǁLockInModelǁpredict_adoption_rate__mutmut_5, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_6': xǁLockInModelǁpredict_adoption_rate__mutmut_6, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_7': xǁLockInModelǁpredict_adoption_rate__mutmut_7, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_8': xǁLockInModelǁpredict_adoption_rate__mutmut_8, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_9': xǁLockInModelǁpredict_adoption_rate__mutmut_9, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_10': xǁLockInModelǁpredict_adoption_rate__mutmut_10, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_11': xǁLockInModelǁpredict_adoption_rate__mutmut_11, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_12': xǁLockInModelǁpredict_adoption_rate__mutmut_12, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_13': xǁLockInModelǁpredict_adoption_rate__mutmut_13, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_14': xǁLockInModelǁpredict_adoption_rate__mutmut_14, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_15': xǁLockInModelǁpredict_adoption_rate__mutmut_15, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_16': xǁLockInModelǁpredict_adoption_rate__mutmut_16, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_17': xǁLockInModelǁpredict_adoption_rate__mutmut_17, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_18': xǁLockInModelǁpredict_adoption_rate__mutmut_18, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_19': xǁLockInModelǁpredict_adoption_rate__mutmut_19, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_20': xǁLockInModelǁpredict_adoption_rate__mutmut_20, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_21': xǁLockInModelǁpredict_adoption_rate__mutmut_21, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_22': xǁLockInModelǁpredict_adoption_rate__mutmut_22, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_23': xǁLockInModelǁpredict_adoption_rate__mutmut_23, 
        'xǁLockInModelǁpredict_adoption_rate__mutmut_24': xǁLockInModelǁpredict_adoption_rate__mutmut_24
    }
    xǁLockInModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁLockInModelǁpredict_adoption_rate'
