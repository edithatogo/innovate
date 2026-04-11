from collections.abc import Sequence
from typing import Any

import numpy as np
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


class ComplementaryGoodsModel:
    """A model for the diffusion of two complementary goods, where the
    adoption of each good is positively influenced by the adoption of the
    other.
    """

    def __init__(self) -> None:
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁComplementaryGoodsModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁComplementaryGoodsModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁComplementaryGoodsModelǁ__init____mutmut_orig(self) -> None:
        self._params: dict[str, float] = {}

    def xǁComplementaryGoodsModelǁ__init____mutmut_1(self) -> None:
        self._params: dict[str, float] = None
    
    xǁComplementaryGoodsModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁComplementaryGoodsModelǁ__init____mutmut_1': xǁComplementaryGoodsModelǁ__init____mutmut_1
    }
    xǁComplementaryGoodsModelǁ__init____mutmut_orig.__name__ = 'xǁComplementaryGoodsModelǁ__init__'

    @property
    def param_names(self) -> Sequence[str]:
        return [
            "k1",  # Intrinsic growth rate of good 1
            "k2",  # Intrinsic growth rate of good 2
            "c1",  # Influence of good 2 on good 1
            "c2",  # Influence of good 1 on good 2
        ]

    def differential_equation(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        args = [y, t, *params]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_orig'), object.__getattribute__(self, 'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_mutants'), args, kwargs, self)

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_orig(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_1(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = None
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_2(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = None
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_3(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = None
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_4(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) - c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_5(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 / (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_6(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 / y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_7(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 + y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_8(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (2 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_9(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 / y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_10(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 / y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_11(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = None
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_12(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) - c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_13(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 / (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_14(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 / y2 * (1 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_15(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 + y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_16(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (2 - y2) + c2 * y1 * y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_17(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 / y2
        return [dy1_dt, dy2_dt]

    def xǁComplementaryGoodsModelǁdifferential_equation__mutmut_18(self, y: np.ndarray, t: float, *params: float) -> Sequence[float]:
        y1, y2 = y
        k1, k2, c1, c2 = params
        dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
        dy2_dt = k2 * y2 * (1 - y2) + c2 / y1 * y2
        return [dy1_dt, dy2_dt]
    
    xǁComplementaryGoodsModelǁdifferential_equation__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_1': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_1, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_2': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_2, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_3': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_3, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_4': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_4, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_5': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_5, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_6': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_6, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_7': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_7, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_8': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_8, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_9': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_9, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_10': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_10, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_11': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_11, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_12': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_12, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_13': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_13, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_14': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_14, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_15': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_15, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_16': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_16, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_17': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_17, 
        'xǁComplementaryGoodsModelǁdifferential_equation__mutmut_18': xǁComplementaryGoodsModelǁdifferential_equation__mutmut_18
    }
    xǁComplementaryGoodsModelǁdifferential_equation__mutmut_orig.__name__ = 'xǁComplementaryGoodsModelǁdifferential_equation'

    def predict(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        args = [t, y0]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁComplementaryGoodsModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁComplementaryGoodsModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁComplementaryGoodsModelǁpredict__mutmut_orig(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, t, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_1(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, t, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_2(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError(None)

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, t, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_3(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("XXModel parameters have not been set.XX")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, t, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_4(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, t, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_5(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("MODEL PARAMETERS HAVE NOT BEEN SET.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, t, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_6(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = None
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_7(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(None, y0, t, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_8(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, None, t, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_9(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, None, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_10(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, t, args=None)
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_11(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(y0, t, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_12(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, t, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_13(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, args=tuple(self._params.values()))
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_14(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, t, )
        return solution

    def xǁComplementaryGoodsModelǁpredict__mutmut_15(self, t: Sequence[float], y0: np.ndarray) -> np.ndarray:
        """Predicts the adoption of both goods over time."""
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        solution = odeint(self.differential_equation, y0, t, args=tuple(None))
        return solution
    
    xǁComplementaryGoodsModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁComplementaryGoodsModelǁpredict__mutmut_1': xǁComplementaryGoodsModelǁpredict__mutmut_1, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_2': xǁComplementaryGoodsModelǁpredict__mutmut_2, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_3': xǁComplementaryGoodsModelǁpredict__mutmut_3, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_4': xǁComplementaryGoodsModelǁpredict__mutmut_4, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_5': xǁComplementaryGoodsModelǁpredict__mutmut_5, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_6': xǁComplementaryGoodsModelǁpredict__mutmut_6, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_7': xǁComplementaryGoodsModelǁpredict__mutmut_7, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_8': xǁComplementaryGoodsModelǁpredict__mutmut_8, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_9': xǁComplementaryGoodsModelǁpredict__mutmut_9, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_10': xǁComplementaryGoodsModelǁpredict__mutmut_10, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_11': xǁComplementaryGoodsModelǁpredict__mutmut_11, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_12': xǁComplementaryGoodsModelǁpredict__mutmut_12, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_13': xǁComplementaryGoodsModelǁpredict__mutmut_13, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_14': xǁComplementaryGoodsModelǁpredict__mutmut_14, 
        'xǁComplementaryGoodsModelǁpredict__mutmut_15': xǁComplementaryGoodsModelǁpredict__mutmut_15
    }
    xǁComplementaryGoodsModelǁpredict__mutmut_orig.__name__ = 'xǁComplementaryGoodsModelǁpredict'

    def fit(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        args = [t, y]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁComplementaryGoodsModelǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁComplementaryGoodsModelǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁComplementaryGoodsModelǁfit__mutmut_orig(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_1(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = None
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_2(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(None)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_3(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 and y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_4(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim == 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_5(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 3 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_6(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[2] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_7(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] == 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_8(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 3:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_9(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(None)

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_10(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("XX`y` must be a 2D array with two columns.XX")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_11(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2d array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_12(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`Y` MUST BE A 2D ARRAY WITH TWO COLUMNS.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_13(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = None

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_14(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[1, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_15(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = None
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_16(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(None)
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_17(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(None, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_18(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, None))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_19(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_20(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, ))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_21(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = None
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_22(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(None, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_23(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, None)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_24(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_25(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, )
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_26(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
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

    def xǁComplementaryGoodsModelǁfit__mutmut_27(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
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

    def xǁComplementaryGoodsModelǁfit__mutmut_28(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) * 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_29(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y + y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_30(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 3))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_31(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_32(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_33(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_34(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_35(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_36(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_37(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_38(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_39(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_40(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_41(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_42(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_43(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = None

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def xǁComplementaryGoodsModelǁfit__mutmut_44(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_45(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_46(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_47(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_48(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_49(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_50(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_51(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_52(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_53(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_54(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_55(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_56(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_57(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_58(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_59(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_60(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_61(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_62(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_63(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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

    def xǁComplementaryGoodsModelǁfit__mutmut_64(self, t: Sequence[float], y: np.ndarray, **kwargs: Any) -> "ComplementaryGoodsModel":
        """Fits the model to the data."""
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params: np.ndarray, t: Sequence[float], y: np.ndarray) -> float:
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return float(np.sum((y - y_pred) ** 2))

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
    
    xǁComplementaryGoodsModelǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁComplementaryGoodsModelǁfit__mutmut_1': xǁComplementaryGoodsModelǁfit__mutmut_1, 
        'xǁComplementaryGoodsModelǁfit__mutmut_2': xǁComplementaryGoodsModelǁfit__mutmut_2, 
        'xǁComplementaryGoodsModelǁfit__mutmut_3': xǁComplementaryGoodsModelǁfit__mutmut_3, 
        'xǁComplementaryGoodsModelǁfit__mutmut_4': xǁComplementaryGoodsModelǁfit__mutmut_4, 
        'xǁComplementaryGoodsModelǁfit__mutmut_5': xǁComplementaryGoodsModelǁfit__mutmut_5, 
        'xǁComplementaryGoodsModelǁfit__mutmut_6': xǁComplementaryGoodsModelǁfit__mutmut_6, 
        'xǁComplementaryGoodsModelǁfit__mutmut_7': xǁComplementaryGoodsModelǁfit__mutmut_7, 
        'xǁComplementaryGoodsModelǁfit__mutmut_8': xǁComplementaryGoodsModelǁfit__mutmut_8, 
        'xǁComplementaryGoodsModelǁfit__mutmut_9': xǁComplementaryGoodsModelǁfit__mutmut_9, 
        'xǁComplementaryGoodsModelǁfit__mutmut_10': xǁComplementaryGoodsModelǁfit__mutmut_10, 
        'xǁComplementaryGoodsModelǁfit__mutmut_11': xǁComplementaryGoodsModelǁfit__mutmut_11, 
        'xǁComplementaryGoodsModelǁfit__mutmut_12': xǁComplementaryGoodsModelǁfit__mutmut_12, 
        'xǁComplementaryGoodsModelǁfit__mutmut_13': xǁComplementaryGoodsModelǁfit__mutmut_13, 
        'xǁComplementaryGoodsModelǁfit__mutmut_14': xǁComplementaryGoodsModelǁfit__mutmut_14, 
        'xǁComplementaryGoodsModelǁfit__mutmut_15': xǁComplementaryGoodsModelǁfit__mutmut_15, 
        'xǁComplementaryGoodsModelǁfit__mutmut_16': xǁComplementaryGoodsModelǁfit__mutmut_16, 
        'xǁComplementaryGoodsModelǁfit__mutmut_17': xǁComplementaryGoodsModelǁfit__mutmut_17, 
        'xǁComplementaryGoodsModelǁfit__mutmut_18': xǁComplementaryGoodsModelǁfit__mutmut_18, 
        'xǁComplementaryGoodsModelǁfit__mutmut_19': xǁComplementaryGoodsModelǁfit__mutmut_19, 
        'xǁComplementaryGoodsModelǁfit__mutmut_20': xǁComplementaryGoodsModelǁfit__mutmut_20, 
        'xǁComplementaryGoodsModelǁfit__mutmut_21': xǁComplementaryGoodsModelǁfit__mutmut_21, 
        'xǁComplementaryGoodsModelǁfit__mutmut_22': xǁComplementaryGoodsModelǁfit__mutmut_22, 
        'xǁComplementaryGoodsModelǁfit__mutmut_23': xǁComplementaryGoodsModelǁfit__mutmut_23, 
        'xǁComplementaryGoodsModelǁfit__mutmut_24': xǁComplementaryGoodsModelǁfit__mutmut_24, 
        'xǁComplementaryGoodsModelǁfit__mutmut_25': xǁComplementaryGoodsModelǁfit__mutmut_25, 
        'xǁComplementaryGoodsModelǁfit__mutmut_26': xǁComplementaryGoodsModelǁfit__mutmut_26, 
        'xǁComplementaryGoodsModelǁfit__mutmut_27': xǁComplementaryGoodsModelǁfit__mutmut_27, 
        'xǁComplementaryGoodsModelǁfit__mutmut_28': xǁComplementaryGoodsModelǁfit__mutmut_28, 
        'xǁComplementaryGoodsModelǁfit__mutmut_29': xǁComplementaryGoodsModelǁfit__mutmut_29, 
        'xǁComplementaryGoodsModelǁfit__mutmut_30': xǁComplementaryGoodsModelǁfit__mutmut_30, 
        'xǁComplementaryGoodsModelǁfit__mutmut_31': xǁComplementaryGoodsModelǁfit__mutmut_31, 
        'xǁComplementaryGoodsModelǁfit__mutmut_32': xǁComplementaryGoodsModelǁfit__mutmut_32, 
        'xǁComplementaryGoodsModelǁfit__mutmut_33': xǁComplementaryGoodsModelǁfit__mutmut_33, 
        'xǁComplementaryGoodsModelǁfit__mutmut_34': xǁComplementaryGoodsModelǁfit__mutmut_34, 
        'xǁComplementaryGoodsModelǁfit__mutmut_35': xǁComplementaryGoodsModelǁfit__mutmut_35, 
        'xǁComplementaryGoodsModelǁfit__mutmut_36': xǁComplementaryGoodsModelǁfit__mutmut_36, 
        'xǁComplementaryGoodsModelǁfit__mutmut_37': xǁComplementaryGoodsModelǁfit__mutmut_37, 
        'xǁComplementaryGoodsModelǁfit__mutmut_38': xǁComplementaryGoodsModelǁfit__mutmut_38, 
        'xǁComplementaryGoodsModelǁfit__mutmut_39': xǁComplementaryGoodsModelǁfit__mutmut_39, 
        'xǁComplementaryGoodsModelǁfit__mutmut_40': xǁComplementaryGoodsModelǁfit__mutmut_40, 
        'xǁComplementaryGoodsModelǁfit__mutmut_41': xǁComplementaryGoodsModelǁfit__mutmut_41, 
        'xǁComplementaryGoodsModelǁfit__mutmut_42': xǁComplementaryGoodsModelǁfit__mutmut_42, 
        'xǁComplementaryGoodsModelǁfit__mutmut_43': xǁComplementaryGoodsModelǁfit__mutmut_43, 
        'xǁComplementaryGoodsModelǁfit__mutmut_44': xǁComplementaryGoodsModelǁfit__mutmut_44, 
        'xǁComplementaryGoodsModelǁfit__mutmut_45': xǁComplementaryGoodsModelǁfit__mutmut_45, 
        'xǁComplementaryGoodsModelǁfit__mutmut_46': xǁComplementaryGoodsModelǁfit__mutmut_46, 
        'xǁComplementaryGoodsModelǁfit__mutmut_47': xǁComplementaryGoodsModelǁfit__mutmut_47, 
        'xǁComplementaryGoodsModelǁfit__mutmut_48': xǁComplementaryGoodsModelǁfit__mutmut_48, 
        'xǁComplementaryGoodsModelǁfit__mutmut_49': xǁComplementaryGoodsModelǁfit__mutmut_49, 
        'xǁComplementaryGoodsModelǁfit__mutmut_50': xǁComplementaryGoodsModelǁfit__mutmut_50, 
        'xǁComplementaryGoodsModelǁfit__mutmut_51': xǁComplementaryGoodsModelǁfit__mutmut_51, 
        'xǁComplementaryGoodsModelǁfit__mutmut_52': xǁComplementaryGoodsModelǁfit__mutmut_52, 
        'xǁComplementaryGoodsModelǁfit__mutmut_53': xǁComplementaryGoodsModelǁfit__mutmut_53, 
        'xǁComplementaryGoodsModelǁfit__mutmut_54': xǁComplementaryGoodsModelǁfit__mutmut_54, 
        'xǁComplementaryGoodsModelǁfit__mutmut_55': xǁComplementaryGoodsModelǁfit__mutmut_55, 
        'xǁComplementaryGoodsModelǁfit__mutmut_56': xǁComplementaryGoodsModelǁfit__mutmut_56, 
        'xǁComplementaryGoodsModelǁfit__mutmut_57': xǁComplementaryGoodsModelǁfit__mutmut_57, 
        'xǁComplementaryGoodsModelǁfit__mutmut_58': xǁComplementaryGoodsModelǁfit__mutmut_58, 
        'xǁComplementaryGoodsModelǁfit__mutmut_59': xǁComplementaryGoodsModelǁfit__mutmut_59, 
        'xǁComplementaryGoodsModelǁfit__mutmut_60': xǁComplementaryGoodsModelǁfit__mutmut_60, 
        'xǁComplementaryGoodsModelǁfit__mutmut_61': xǁComplementaryGoodsModelǁfit__mutmut_61, 
        'xǁComplementaryGoodsModelǁfit__mutmut_62': xǁComplementaryGoodsModelǁfit__mutmut_62, 
        'xǁComplementaryGoodsModelǁfit__mutmut_63': xǁComplementaryGoodsModelǁfit__mutmut_63, 
        'xǁComplementaryGoodsModelǁfit__mutmut_64': xǁComplementaryGoodsModelǁfit__mutmut_64
    }
    xǁComplementaryGoodsModelǁfit__mutmut_orig.__name__ = 'xǁComplementaryGoodsModelǁfit'

    def initial_guesses(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_orig'), object.__getattribute__(self, 'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_mutants'), args, kwargs, self)

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_orig(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_1(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) <= 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_2(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 3:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_3(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"XXk1XX": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_4(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"K1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_5(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 1.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_6(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "XXk2XX": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_7(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "K2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_8(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 1.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_9(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "XXc1XX": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_10(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "C1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_11(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 1.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_12(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "XXc2XX": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_13(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "C2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_14(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 1.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_15(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = None
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_16(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(None, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_17(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, None)
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_18(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_19(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, )
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_20(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(6, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_21(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = None
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_22(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(None)
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_23(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = None

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_24(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide=None, invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_25(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid=None):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_26(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_27(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", ):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_28(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="XXignoreXX", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_29(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="IGNORE", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_30(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="XXignoreXX"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_31(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="IGNORE"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_32(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = None
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_33(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                None,
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_34(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) * t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_35(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(None) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_36(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] * y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_37(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[2:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_38(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_39(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[1, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_40(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 1]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_41(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[2:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_42(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = None

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_43(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                None,
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_44(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) * t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_45(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(None) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_46(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] * y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_47(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[2:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_48(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 2] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_49(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[1, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_50(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 2]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_51(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[2:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_52(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = None
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_53(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) or k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_54(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(None) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_55(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est >= 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_56(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 1 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_57(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 1.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_58(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = None

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_59(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) or k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_60(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(None) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_61(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est >= 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_62(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 1 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_63(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 1.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_64(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"XXk1XX": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_65(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"K1": k1, "k2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_66(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "XXk2XX": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_67(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "K2": k2, "c1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_68(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "XXc1XX": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_69(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "C1": 0.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_70(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 1.01, "c2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_71(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "XXc2XX": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_72(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "C2": 0.01}

    def xǁComplementaryGoodsModelǁinitial_guesses__mutmut_73(self, t: Sequence[float], y: np.ndarray) -> dict[str, float]:
        # A simple heuristic for initial guesses
        if len(t) < 2:
            return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

        # Use the first few data points to estimate initial growth
        num_initial_points = min(5, len(t))
        t_initial = np.array(t[:num_initial_points])
        y_initial = y[:num_initial_points]

        # Estimate k1 and k2 from the initial exponential growth
        # y(t) ~= y(0) * exp(k*t) => k ~= log(y(t)/y(0)) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            k1_est = np.nanmean(
                np.log(y_initial[1:, 0] / y_initial[0, 0]) / t_initial[1:],
            )
            k2_est = np.nanmean(
                np.log(y_initial[1:, 1] / y_initial[0, 1]) / t_initial[1:],
            )

        k1 = k1_est if np.isfinite(k1_est) and k1_est > 0 else 0.1
        k2 = k2_est if np.isfinite(k2_est) and k2_est > 0 else 0.1

        # For c1 and c2, we can start with small positive values
        return {"k1": k1, "k2": k2, "c1": 0.01, "c2": 1.01}
    
    xǁComplementaryGoodsModelǁinitial_guesses__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_1': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_1, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_2': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_2, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_3': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_3, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_4': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_4, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_5': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_5, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_6': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_6, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_7': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_7, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_8': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_8, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_9': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_9, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_10': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_10, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_11': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_11, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_12': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_12, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_13': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_13, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_14': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_14, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_15': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_15, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_16': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_16, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_17': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_17, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_18': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_18, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_19': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_19, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_20': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_20, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_21': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_21, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_22': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_22, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_23': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_23, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_24': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_24, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_25': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_25, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_26': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_26, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_27': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_27, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_28': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_28, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_29': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_29, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_30': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_30, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_31': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_31, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_32': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_32, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_33': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_33, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_34': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_34, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_35': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_35, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_36': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_36, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_37': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_37, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_38': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_38, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_39': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_39, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_40': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_40, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_41': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_41, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_42': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_42, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_43': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_43, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_44': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_44, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_45': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_45, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_46': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_46, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_47': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_47, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_48': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_48, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_49': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_49, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_50': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_50, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_51': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_51, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_52': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_52, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_53': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_53, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_54': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_54, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_55': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_55, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_56': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_56, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_57': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_57, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_58': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_58, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_59': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_59, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_60': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_60, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_61': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_61, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_62': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_62, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_63': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_63, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_64': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_64, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_65': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_65, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_66': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_66, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_67': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_67, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_68': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_68, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_69': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_69, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_70': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_70, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_71': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_71, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_72': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_72, 
        'xǁComplementaryGoodsModelǁinitial_guesses__mutmut_73': xǁComplementaryGoodsModelǁinitial_guesses__mutmut_73
    }
    xǁComplementaryGoodsModelǁinitial_guesses__mutmut_orig.__name__ = 'xǁComplementaryGoodsModelǁinitial_guesses'

    def bounds(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁComplementaryGoodsModelǁbounds__mutmut_orig'), object.__getattribute__(self, 'xǁComplementaryGoodsModelǁbounds__mutmut_mutants'), args, kwargs, self)

    def xǁComplementaryGoodsModelǁbounds__mutmut_orig(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (0, np.inf),
            "k2": (0, np.inf),
            "c1": (0, np.inf),
            "c2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_1(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "XXk1XX": (0, np.inf),
            "k2": (0, np.inf),
            "c1": (0, np.inf),
            "c2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_2(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "K1": (0, np.inf),
            "k2": (0, np.inf),
            "c1": (0, np.inf),
            "c2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_3(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (1, np.inf),
            "k2": (0, np.inf),
            "c1": (0, np.inf),
            "c2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_4(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (0, np.inf),
            "XXk2XX": (0, np.inf),
            "c1": (0, np.inf),
            "c2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_5(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (0, np.inf),
            "K2": (0, np.inf),
            "c1": (0, np.inf),
            "c2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_6(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (0, np.inf),
            "k2": (1, np.inf),
            "c1": (0, np.inf),
            "c2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_7(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (0, np.inf),
            "k2": (0, np.inf),
            "XXc1XX": (0, np.inf),
            "c2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_8(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (0, np.inf),
            "k2": (0, np.inf),
            "C1": (0, np.inf),
            "c2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_9(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (0, np.inf),
            "k2": (0, np.inf),
            "c1": (1, np.inf),
            "c2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_10(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (0, np.inf),
            "k2": (0, np.inf),
            "c1": (0, np.inf),
            "XXc2XX": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_11(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (0, np.inf),
            "k2": (0, np.inf),
            "c1": (0, np.inf),
            "C2": (0, np.inf),
        }

    def xǁComplementaryGoodsModelǁbounds__mutmut_12(self, t: Sequence[float], y: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "k1": (0, np.inf),
            "k2": (0, np.inf),
            "c1": (0, np.inf),
            "c2": (1, np.inf),
        }
    
    xǁComplementaryGoodsModelǁbounds__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁComplementaryGoodsModelǁbounds__mutmut_1': xǁComplementaryGoodsModelǁbounds__mutmut_1, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_2': xǁComplementaryGoodsModelǁbounds__mutmut_2, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_3': xǁComplementaryGoodsModelǁbounds__mutmut_3, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_4': xǁComplementaryGoodsModelǁbounds__mutmut_4, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_5': xǁComplementaryGoodsModelǁbounds__mutmut_5, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_6': xǁComplementaryGoodsModelǁbounds__mutmut_6, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_7': xǁComplementaryGoodsModelǁbounds__mutmut_7, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_8': xǁComplementaryGoodsModelǁbounds__mutmut_8, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_9': xǁComplementaryGoodsModelǁbounds__mutmut_9, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_10': xǁComplementaryGoodsModelǁbounds__mutmut_10, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_11': xǁComplementaryGoodsModelǁbounds__mutmut_11, 
        'xǁComplementaryGoodsModelǁbounds__mutmut_12': xǁComplementaryGoodsModelǁbounds__mutmut_12
    }
    xǁComplementaryGoodsModelǁbounds__mutmut_orig.__name__ = 'xǁComplementaryGoodsModelǁbounds'

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]) -> None:
        self._params = value

    def score(self, t: Sequence[float], y: np.ndarray) -> float:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁComplementaryGoodsModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁComplementaryGoodsModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁComplementaryGoodsModelǁscore__mutmut_orig(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_1(self, t: Sequence[float], y: np.ndarray) -> float:
        if self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_2(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError(None)
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_3(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet.XX")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_4(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_5(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_6(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = None
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_7(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(None, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_8(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, None)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_9(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_10(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, )
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_11(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[1, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_12(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = None
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_13(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum(None)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_14(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) * 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_15(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y + y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_16(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 3)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_17(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = None
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_18(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum(None)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_19(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) * 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_20(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y + np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_21(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(None, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_22(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=None)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_23(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_24(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, )) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_25(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=1)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_26(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 3)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_27(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_28(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_29(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_30(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_31(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0

    def xǁComplementaryGoodsModelǁscore__mutmut_32(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    xǁComplementaryGoodsModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁComplementaryGoodsModelǁscore__mutmut_1': xǁComplementaryGoodsModelǁscore__mutmut_1, 
        'xǁComplementaryGoodsModelǁscore__mutmut_2': xǁComplementaryGoodsModelǁscore__mutmut_2, 
        'xǁComplementaryGoodsModelǁscore__mutmut_3': xǁComplementaryGoodsModelǁscore__mutmut_3, 
        'xǁComplementaryGoodsModelǁscore__mutmut_4': xǁComplementaryGoodsModelǁscore__mutmut_4, 
        'xǁComplementaryGoodsModelǁscore__mutmut_5': xǁComplementaryGoodsModelǁscore__mutmut_5, 
        'xǁComplementaryGoodsModelǁscore__mutmut_6': xǁComplementaryGoodsModelǁscore__mutmut_6, 
        'xǁComplementaryGoodsModelǁscore__mutmut_7': xǁComplementaryGoodsModelǁscore__mutmut_7, 
        'xǁComplementaryGoodsModelǁscore__mutmut_8': xǁComplementaryGoodsModelǁscore__mutmut_8, 
        'xǁComplementaryGoodsModelǁscore__mutmut_9': xǁComplementaryGoodsModelǁscore__mutmut_9, 
        'xǁComplementaryGoodsModelǁscore__mutmut_10': xǁComplementaryGoodsModelǁscore__mutmut_10, 
        'xǁComplementaryGoodsModelǁscore__mutmut_11': xǁComplementaryGoodsModelǁscore__mutmut_11, 
        'xǁComplementaryGoodsModelǁscore__mutmut_12': xǁComplementaryGoodsModelǁscore__mutmut_12, 
        'xǁComplementaryGoodsModelǁscore__mutmut_13': xǁComplementaryGoodsModelǁscore__mutmut_13, 
        'xǁComplementaryGoodsModelǁscore__mutmut_14': xǁComplementaryGoodsModelǁscore__mutmut_14, 
        'xǁComplementaryGoodsModelǁscore__mutmut_15': xǁComplementaryGoodsModelǁscore__mutmut_15, 
        'xǁComplementaryGoodsModelǁscore__mutmut_16': xǁComplementaryGoodsModelǁscore__mutmut_16, 
        'xǁComplementaryGoodsModelǁscore__mutmut_17': xǁComplementaryGoodsModelǁscore__mutmut_17, 
        'xǁComplementaryGoodsModelǁscore__mutmut_18': xǁComplementaryGoodsModelǁscore__mutmut_18, 
        'xǁComplementaryGoodsModelǁscore__mutmut_19': xǁComplementaryGoodsModelǁscore__mutmut_19, 
        'xǁComplementaryGoodsModelǁscore__mutmut_20': xǁComplementaryGoodsModelǁscore__mutmut_20, 
        'xǁComplementaryGoodsModelǁscore__mutmut_21': xǁComplementaryGoodsModelǁscore__mutmut_21, 
        'xǁComplementaryGoodsModelǁscore__mutmut_22': xǁComplementaryGoodsModelǁscore__mutmut_22, 
        'xǁComplementaryGoodsModelǁscore__mutmut_23': xǁComplementaryGoodsModelǁscore__mutmut_23, 
        'xǁComplementaryGoodsModelǁscore__mutmut_24': xǁComplementaryGoodsModelǁscore__mutmut_24, 
        'xǁComplementaryGoodsModelǁscore__mutmut_25': xǁComplementaryGoodsModelǁscore__mutmut_25, 
        'xǁComplementaryGoodsModelǁscore__mutmut_26': xǁComplementaryGoodsModelǁscore__mutmut_26, 
        'xǁComplementaryGoodsModelǁscore__mutmut_27': xǁComplementaryGoodsModelǁscore__mutmut_27, 
        'xǁComplementaryGoodsModelǁscore__mutmut_28': xǁComplementaryGoodsModelǁscore__mutmut_28, 
        'xǁComplementaryGoodsModelǁscore__mutmut_29': xǁComplementaryGoodsModelǁscore__mutmut_29, 
        'xǁComplementaryGoodsModelǁscore__mutmut_30': xǁComplementaryGoodsModelǁscore__mutmut_30, 
        'xǁComplementaryGoodsModelǁscore__mutmut_31': xǁComplementaryGoodsModelǁscore__mutmut_31, 
        'xǁComplementaryGoodsModelǁscore__mutmut_32': xǁComplementaryGoodsModelǁscore__mutmut_32
    }
    xǁComplementaryGoodsModelǁscore__mutmut_orig.__name__ = 'xǁComplementaryGoodsModelǁscore'

    def predict_adoption_rate(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        args = [t, y0]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_orig(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_1(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_2(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError(None)
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_3(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet.XX")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_4(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_5(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_6(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = None
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_7(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(None, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_8(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, None)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_9(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_10(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, )
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_11(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = None
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_12(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["XXk1XX"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_13(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["K1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_14(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["XXk2XX"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_15(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["K2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_16(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["XXc1XX"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_17(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["C1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_18(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["XXc2XX"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_19(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["C2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_20(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = None
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_21(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) - c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_22(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] / (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_23(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 / y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_24(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 1] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_25(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 + y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_26(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (2 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_27(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 1]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_28(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] / y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_29(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 / y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_30(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 1] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_31(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 2]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_32(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = None
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_33(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) - c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_34(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] / (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_35(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 / y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_36(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 2] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_37(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 + y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_38(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (2 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_39(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 2]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_40(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] / y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_41(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 / y_pred[:, 0] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_42(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 1] * y_pred[:, 1]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_43(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 2]
        return np.vstack([dy1_dt, dy2_dt]).T

    def xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_44(
        self,
        t: Sequence[float],
        y0: np.ndarray,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y0)
        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )
        dy1_dt = k1 * y_pred[:, 0] * (1 - y_pred[:, 0]) + c1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = k2 * y_pred[:, 1] * (1 - y_pred[:, 1]) + c2 * y_pred[:, 0] * y_pred[:, 1]
        return np.vstack(None).T
    
    xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_1': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_1, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_2': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_2, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_3': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_3, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_4': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_4, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_5': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_5, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_6': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_6, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_7': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_7, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_8': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_8, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_9': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_9, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_10': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_10, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_11': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_11, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_12': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_12, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_13': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_13, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_14': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_14, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_15': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_15, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_16': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_16, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_17': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_17, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_18': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_18, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_19': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_19, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_20': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_20, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_21': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_21, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_22': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_22, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_23': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_23, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_24': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_24, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_25': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_25, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_26': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_26, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_27': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_27, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_28': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_28, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_29': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_29, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_30': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_30, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_31': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_31, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_32': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_32, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_33': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_33, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_34': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_34, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_35': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_35, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_36': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_36, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_37': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_37, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_38': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_38, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_39': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_39, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_40': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_40, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_41': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_41, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_42': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_42, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_43': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_43, 
        'xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_44': xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_44
    }
    xǁComplementaryGoodsModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁComplementaryGoodsModelǁpredict_adoption_rate'
