import numpy as np

from .base import ContagionSpread
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


class SIR(ContagionSpread):
    """Implements the Susceptible-Infected-Recovered (SIR) model."""

    def __init__(self, beta: float = 0.2, gamma: float = 0.1):
        args = [beta, gamma]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSIRǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁSIRǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁSIRǁ__init____mutmut_orig(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = gamma

    def xǁSIRǁ__init____mutmut_1(self, beta: float = 1.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = gamma

    def xǁSIRǁ__init____mutmut_2(self, beta: float = 0.2, gamma: float = 1.1):
        self.beta = beta
        self.gamma = gamma

    def xǁSIRǁ__init____mutmut_3(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = None
        self.gamma = gamma

    def xǁSIRǁ__init____mutmut_4(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = None
    
    xǁSIRǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSIRǁ__init____mutmut_1': xǁSIRǁ__init____mutmut_1, 
        'xǁSIRǁ__init____mutmut_2': xǁSIRǁ__init____mutmut_2, 
        'xǁSIRǁ__init____mutmut_3': xǁSIRǁ__init____mutmut_3, 
        'xǁSIRǁ__init____mutmut_4': xǁSIRǁ__init____mutmut_4
    }
    xǁSIRǁ__init____mutmut_orig.__name__ = 'xǁSIRǁ__init__'

    def differential(self, y: np.ndarray, t: float) -> np.ndarray:
        args = [y, t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSIRǁdifferential__mutmut_orig'), object.__getattribute__(self, 'xǁSIRǁdifferential__mutmut_mutants'), args, kwargs, self)

    def xǁSIRǁdifferential__mutmut_orig(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_1(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = None
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_2(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = None
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_3(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S / I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_4(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta / S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_5(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = +self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_6(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = None
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_7(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I + self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_8(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S / I - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_9(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta / S * I - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_10(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma / I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_11(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = None
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_12(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma / I
        return np.array([dSdt, dIdt, dRdt])

    def xǁSIRǁdifferential__mutmut_13(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return np.array(None)
    
    xǁSIRǁdifferential__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSIRǁdifferential__mutmut_1': xǁSIRǁdifferential__mutmut_1, 
        'xǁSIRǁdifferential__mutmut_2': xǁSIRǁdifferential__mutmut_2, 
        'xǁSIRǁdifferential__mutmut_3': xǁSIRǁdifferential__mutmut_3, 
        'xǁSIRǁdifferential__mutmut_4': xǁSIRǁdifferential__mutmut_4, 
        'xǁSIRǁdifferential__mutmut_5': xǁSIRǁdifferential__mutmut_5, 
        'xǁSIRǁdifferential__mutmut_6': xǁSIRǁdifferential__mutmut_6, 
        'xǁSIRǁdifferential__mutmut_7': xǁSIRǁdifferential__mutmut_7, 
        'xǁSIRǁdifferential__mutmut_8': xǁSIRǁdifferential__mutmut_8, 
        'xǁSIRǁdifferential__mutmut_9': xǁSIRǁdifferential__mutmut_9, 
        'xǁSIRǁdifferential__mutmut_10': xǁSIRǁdifferential__mutmut_10, 
        'xǁSIRǁdifferential__mutmut_11': xǁSIRǁdifferential__mutmut_11, 
        'xǁSIRǁdifferential__mutmut_12': xǁSIRǁdifferential__mutmut_12, 
        'xǁSIRǁdifferential__mutmut_13': xǁSIRǁdifferential__mutmut_13
    }
    xǁSIRǁdifferential__mutmut_orig.__name__ = 'xǁSIRǁdifferential'

    def compute_spread_rate(self, **params):
        args = []# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSIRǁcompute_spread_rate__mutmut_orig'), object.__getattribute__(self, 'xǁSIRǁcompute_spread_rate__mutmut_mutants'), args, kwargs, self)

    def xǁSIRǁcompute_spread_rate__mutmut_orig(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_1(self, **params):
        """Calculates the instantaneous spread rate."""
        S = None
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_2(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get(None)
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_3(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("XXSXX")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_4(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("s")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_5(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = None
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_6(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get(None)
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_7(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("XXIXX")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_8(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("i")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_9(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = None
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_10(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get(None, self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_11(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", None)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_12(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get(self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_13(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", )
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_14(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("XXtransmission_rateXX", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_15(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("TRANSMISSION_RATE", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_16(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = None

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_17(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get(None, self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_18(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", None)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_19(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get(self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_20(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", )

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_21(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("XXrecovery_rateXX", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_22(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("RECOVERY_RATE", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_23(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = None
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_24(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S / I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_25(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta / S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_26(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = +beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_27(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = None
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_28(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I + gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_29(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S / I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_30(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta / S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_31(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma / I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_32(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = None
        return dSdt, dIdt, dRdt

    def xǁSIRǁcompute_spread_rate__mutmut_33(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma / I
        return dSdt, dIdt, dRdt
    
    xǁSIRǁcompute_spread_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSIRǁcompute_spread_rate__mutmut_1': xǁSIRǁcompute_spread_rate__mutmut_1, 
        'xǁSIRǁcompute_spread_rate__mutmut_2': xǁSIRǁcompute_spread_rate__mutmut_2, 
        'xǁSIRǁcompute_spread_rate__mutmut_3': xǁSIRǁcompute_spread_rate__mutmut_3, 
        'xǁSIRǁcompute_spread_rate__mutmut_4': xǁSIRǁcompute_spread_rate__mutmut_4, 
        'xǁSIRǁcompute_spread_rate__mutmut_5': xǁSIRǁcompute_spread_rate__mutmut_5, 
        'xǁSIRǁcompute_spread_rate__mutmut_6': xǁSIRǁcompute_spread_rate__mutmut_6, 
        'xǁSIRǁcompute_spread_rate__mutmut_7': xǁSIRǁcompute_spread_rate__mutmut_7, 
        'xǁSIRǁcompute_spread_rate__mutmut_8': xǁSIRǁcompute_spread_rate__mutmut_8, 
        'xǁSIRǁcompute_spread_rate__mutmut_9': xǁSIRǁcompute_spread_rate__mutmut_9, 
        'xǁSIRǁcompute_spread_rate__mutmut_10': xǁSIRǁcompute_spread_rate__mutmut_10, 
        'xǁSIRǁcompute_spread_rate__mutmut_11': xǁSIRǁcompute_spread_rate__mutmut_11, 
        'xǁSIRǁcompute_spread_rate__mutmut_12': xǁSIRǁcompute_spread_rate__mutmut_12, 
        'xǁSIRǁcompute_spread_rate__mutmut_13': xǁSIRǁcompute_spread_rate__mutmut_13, 
        'xǁSIRǁcompute_spread_rate__mutmut_14': xǁSIRǁcompute_spread_rate__mutmut_14, 
        'xǁSIRǁcompute_spread_rate__mutmut_15': xǁSIRǁcompute_spread_rate__mutmut_15, 
        'xǁSIRǁcompute_spread_rate__mutmut_16': xǁSIRǁcompute_spread_rate__mutmut_16, 
        'xǁSIRǁcompute_spread_rate__mutmut_17': xǁSIRǁcompute_spread_rate__mutmut_17, 
        'xǁSIRǁcompute_spread_rate__mutmut_18': xǁSIRǁcompute_spread_rate__mutmut_18, 
        'xǁSIRǁcompute_spread_rate__mutmut_19': xǁSIRǁcompute_spread_rate__mutmut_19, 
        'xǁSIRǁcompute_spread_rate__mutmut_20': xǁSIRǁcompute_spread_rate__mutmut_20, 
        'xǁSIRǁcompute_spread_rate__mutmut_21': xǁSIRǁcompute_spread_rate__mutmut_21, 
        'xǁSIRǁcompute_spread_rate__mutmut_22': xǁSIRǁcompute_spread_rate__mutmut_22, 
        'xǁSIRǁcompute_spread_rate__mutmut_23': xǁSIRǁcompute_spread_rate__mutmut_23, 
        'xǁSIRǁcompute_spread_rate__mutmut_24': xǁSIRǁcompute_spread_rate__mutmut_24, 
        'xǁSIRǁcompute_spread_rate__mutmut_25': xǁSIRǁcompute_spread_rate__mutmut_25, 
        'xǁSIRǁcompute_spread_rate__mutmut_26': xǁSIRǁcompute_spread_rate__mutmut_26, 
        'xǁSIRǁcompute_spread_rate__mutmut_27': xǁSIRǁcompute_spread_rate__mutmut_27, 
        'xǁSIRǁcompute_spread_rate__mutmut_28': xǁSIRǁcompute_spread_rate__mutmut_28, 
        'xǁSIRǁcompute_spread_rate__mutmut_29': xǁSIRǁcompute_spread_rate__mutmut_29, 
        'xǁSIRǁcompute_spread_rate__mutmut_30': xǁSIRǁcompute_spread_rate__mutmut_30, 
        'xǁSIRǁcompute_spread_rate__mutmut_31': xǁSIRǁcompute_spread_rate__mutmut_31, 
        'xǁSIRǁcompute_spread_rate__mutmut_32': xǁSIRǁcompute_spread_rate__mutmut_32, 
        'xǁSIRǁcompute_spread_rate__mutmut_33': xǁSIRǁcompute_spread_rate__mutmut_33
    }
    xǁSIRǁcompute_spread_rate__mutmut_orig.__name__ = 'xǁSIRǁcompute_spread_rate'

    def predict_states(self, time_points, **params):
        args = [time_points]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSIRǁpredict_states__mutmut_orig'), object.__getattribute__(self, 'xǁSIRǁpredict_states__mutmut_mutants'), args, kwargs, self)

    def xǁSIRǁpredict_states__mutmut_orig(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_1(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = None
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_2(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get(None, 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_3(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", None)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_4(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get(999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_5(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", )
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_6(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("XXS0XX", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_7(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("s0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_8(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 1000)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_9(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = None
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_10(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get(None, 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_11(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", None)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_12(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get(1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_13(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", )
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_14(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("XXI0XX", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_15(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("i0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_16(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 2)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_17(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = None

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_18(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get(None, 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_19(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", None)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_20(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get(0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_21(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", )

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_22(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("XXR0XX", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_23(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("r0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_24(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_25(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=None, I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_26(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=None, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_27(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_28(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_29(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], )

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_30(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[1], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_31(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_32(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = None
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_33(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            None,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_34(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            None,
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_35(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            None,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_36(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=None,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_37(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method=None,
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_38(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_39(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_40(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_41(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_42(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_43(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[1], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_44(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[+1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_45(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-2]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_46(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="XXLSODAXX",
        )
        return sol.y.T

    def xǁSIRǁpredict_states__mutmut_47(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="lsoda",
        )
        return sol.y.T
    
    xǁSIRǁpredict_states__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSIRǁpredict_states__mutmut_1': xǁSIRǁpredict_states__mutmut_1, 
        'xǁSIRǁpredict_states__mutmut_2': xǁSIRǁpredict_states__mutmut_2, 
        'xǁSIRǁpredict_states__mutmut_3': xǁSIRǁpredict_states__mutmut_3, 
        'xǁSIRǁpredict_states__mutmut_4': xǁSIRǁpredict_states__mutmut_4, 
        'xǁSIRǁpredict_states__mutmut_5': xǁSIRǁpredict_states__mutmut_5, 
        'xǁSIRǁpredict_states__mutmut_6': xǁSIRǁpredict_states__mutmut_6, 
        'xǁSIRǁpredict_states__mutmut_7': xǁSIRǁpredict_states__mutmut_7, 
        'xǁSIRǁpredict_states__mutmut_8': xǁSIRǁpredict_states__mutmut_8, 
        'xǁSIRǁpredict_states__mutmut_9': xǁSIRǁpredict_states__mutmut_9, 
        'xǁSIRǁpredict_states__mutmut_10': xǁSIRǁpredict_states__mutmut_10, 
        'xǁSIRǁpredict_states__mutmut_11': xǁSIRǁpredict_states__mutmut_11, 
        'xǁSIRǁpredict_states__mutmut_12': xǁSIRǁpredict_states__mutmut_12, 
        'xǁSIRǁpredict_states__mutmut_13': xǁSIRǁpredict_states__mutmut_13, 
        'xǁSIRǁpredict_states__mutmut_14': xǁSIRǁpredict_states__mutmut_14, 
        'xǁSIRǁpredict_states__mutmut_15': xǁSIRǁpredict_states__mutmut_15, 
        'xǁSIRǁpredict_states__mutmut_16': xǁSIRǁpredict_states__mutmut_16, 
        'xǁSIRǁpredict_states__mutmut_17': xǁSIRǁpredict_states__mutmut_17, 
        'xǁSIRǁpredict_states__mutmut_18': xǁSIRǁpredict_states__mutmut_18, 
        'xǁSIRǁpredict_states__mutmut_19': xǁSIRǁpredict_states__mutmut_19, 
        'xǁSIRǁpredict_states__mutmut_20': xǁSIRǁpredict_states__mutmut_20, 
        'xǁSIRǁpredict_states__mutmut_21': xǁSIRǁpredict_states__mutmut_21, 
        'xǁSIRǁpredict_states__mutmut_22': xǁSIRǁpredict_states__mutmut_22, 
        'xǁSIRǁpredict_states__mutmut_23': xǁSIRǁpredict_states__mutmut_23, 
        'xǁSIRǁpredict_states__mutmut_24': xǁSIRǁpredict_states__mutmut_24, 
        'xǁSIRǁpredict_states__mutmut_25': xǁSIRǁpredict_states__mutmut_25, 
        'xǁSIRǁpredict_states__mutmut_26': xǁSIRǁpredict_states__mutmut_26, 
        'xǁSIRǁpredict_states__mutmut_27': xǁSIRǁpredict_states__mutmut_27, 
        'xǁSIRǁpredict_states__mutmut_28': xǁSIRǁpredict_states__mutmut_28, 
        'xǁSIRǁpredict_states__mutmut_29': xǁSIRǁpredict_states__mutmut_29, 
        'xǁSIRǁpredict_states__mutmut_30': xǁSIRǁpredict_states__mutmut_30, 
        'xǁSIRǁpredict_states__mutmut_31': xǁSIRǁpredict_states__mutmut_31, 
        'xǁSIRǁpredict_states__mutmut_32': xǁSIRǁpredict_states__mutmut_32, 
        'xǁSIRǁpredict_states__mutmut_33': xǁSIRǁpredict_states__mutmut_33, 
        'xǁSIRǁpredict_states__mutmut_34': xǁSIRǁpredict_states__mutmut_34, 
        'xǁSIRǁpredict_states__mutmut_35': xǁSIRǁpredict_states__mutmut_35, 
        'xǁSIRǁpredict_states__mutmut_36': xǁSIRǁpredict_states__mutmut_36, 
        'xǁSIRǁpredict_states__mutmut_37': xǁSIRǁpredict_states__mutmut_37, 
        'xǁSIRǁpredict_states__mutmut_38': xǁSIRǁpredict_states__mutmut_38, 
        'xǁSIRǁpredict_states__mutmut_39': xǁSIRǁpredict_states__mutmut_39, 
        'xǁSIRǁpredict_states__mutmut_40': xǁSIRǁpredict_states__mutmut_40, 
        'xǁSIRǁpredict_states__mutmut_41': xǁSIRǁpredict_states__mutmut_41, 
        'xǁSIRǁpredict_states__mutmut_42': xǁSIRǁpredict_states__mutmut_42, 
        'xǁSIRǁpredict_states__mutmut_43': xǁSIRǁpredict_states__mutmut_43, 
        'xǁSIRǁpredict_states__mutmut_44': xǁSIRǁpredict_states__mutmut_44, 
        'xǁSIRǁpredict_states__mutmut_45': xǁSIRǁpredict_states__mutmut_45, 
        'xǁSIRǁpredict_states__mutmut_46': xǁSIRǁpredict_states__mutmut_46, 
        'xǁSIRǁpredict_states__mutmut_47': xǁSIRǁpredict_states__mutmut_47
    }
    xǁSIRǁpredict_states__mutmut_orig.__name__ = 'xǁSIRǁpredict_states'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSIRǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁSIRǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁSIRǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters."""
        return {
            "XXtransmission_rateXX": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters."""
        return {
            "TRANSMISSION_RATE": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "XXtypeXX": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "TYPE": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "XXfloatXX",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "FLOAT",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "XXdefaultXX": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "DEFAULT": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "XXdescriptionXX": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "DESCRIPTION": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "XXThe rate of transmission of the contagion.XX",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "the rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "THE RATE OF TRANSMISSION OF THE CONTAGION.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_14(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "XXrecovery_rateXX": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_15(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "RECOVERY_RATE": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_16(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "XXtypeXX": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_17(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "TYPE": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_18(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "XXfloatXX",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_19(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "FLOAT",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_20(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "XXdefaultXX": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_21(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "DEFAULT": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_22(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "XXdescriptionXX": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_23(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "DESCRIPTION": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_24(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "XXThe rate of recovery from the contagion.XX",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_25(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "the rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_26(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "THE RATE OF RECOVERY FROM THE CONTAGION.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_27(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "XXS0XX": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_28(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "s0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_29(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "XXtypeXX": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_30(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "TYPE": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_31(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "XXfloatXX",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_32(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "FLOAT",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_33(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "XXdefaultXX": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_34(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "DEFAULT": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_35(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 1000,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_36(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "XXdescriptionXX": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_37(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "DESCRIPTION": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_38(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "XXThe initial number of susceptible individuals.XX",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_39(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "the initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_40(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "THE INITIAL NUMBER OF SUSCEPTIBLE INDIVIDUALS.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_41(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "XXI0XX": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_42(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "i0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_43(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "XXtypeXX": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_44(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "TYPE": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_45(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "XXfloatXX",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_46(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "FLOAT",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_47(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "XXdefaultXX": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_48(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "DEFAULT": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_49(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 2,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_50(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "XXdescriptionXX": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_51(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "DESCRIPTION": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_52(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "XXThe initial number of infectious individuals.XX",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_53(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "the initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_54(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "THE INITIAL NUMBER OF INFECTIOUS INDIVIDUALS.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_55(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "XXR0XX": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_56(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "r0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_57(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "XXtypeXX": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_58(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "TYPE": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_59(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "XXfloatXX",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_60(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "FLOAT",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_61(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "XXdefaultXX": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_62(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "DEFAULT": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_63(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_64(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "XXdescriptionXX": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_65(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "DESCRIPTION": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_66(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "XXThe initial number of recovered individuals.XX",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_67(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "the initial number of recovered individuals.",
            },
        }

    def xǁSIRǁget_parameters_schema__mutmut_68(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "THE INITIAL NUMBER OF RECOVERED INDIVIDUALS.",
            },
        }
    
    xǁSIRǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSIRǁget_parameters_schema__mutmut_1': xǁSIRǁget_parameters_schema__mutmut_1, 
        'xǁSIRǁget_parameters_schema__mutmut_2': xǁSIRǁget_parameters_schema__mutmut_2, 
        'xǁSIRǁget_parameters_schema__mutmut_3': xǁSIRǁget_parameters_schema__mutmut_3, 
        'xǁSIRǁget_parameters_schema__mutmut_4': xǁSIRǁget_parameters_schema__mutmut_4, 
        'xǁSIRǁget_parameters_schema__mutmut_5': xǁSIRǁget_parameters_schema__mutmut_5, 
        'xǁSIRǁget_parameters_schema__mutmut_6': xǁSIRǁget_parameters_schema__mutmut_6, 
        'xǁSIRǁget_parameters_schema__mutmut_7': xǁSIRǁget_parameters_schema__mutmut_7, 
        'xǁSIRǁget_parameters_schema__mutmut_8': xǁSIRǁget_parameters_schema__mutmut_8, 
        'xǁSIRǁget_parameters_schema__mutmut_9': xǁSIRǁget_parameters_schema__mutmut_9, 
        'xǁSIRǁget_parameters_schema__mutmut_10': xǁSIRǁget_parameters_schema__mutmut_10, 
        'xǁSIRǁget_parameters_schema__mutmut_11': xǁSIRǁget_parameters_schema__mutmut_11, 
        'xǁSIRǁget_parameters_schema__mutmut_12': xǁSIRǁget_parameters_schema__mutmut_12, 
        'xǁSIRǁget_parameters_schema__mutmut_13': xǁSIRǁget_parameters_schema__mutmut_13, 
        'xǁSIRǁget_parameters_schema__mutmut_14': xǁSIRǁget_parameters_schema__mutmut_14, 
        'xǁSIRǁget_parameters_schema__mutmut_15': xǁSIRǁget_parameters_schema__mutmut_15, 
        'xǁSIRǁget_parameters_schema__mutmut_16': xǁSIRǁget_parameters_schema__mutmut_16, 
        'xǁSIRǁget_parameters_schema__mutmut_17': xǁSIRǁget_parameters_schema__mutmut_17, 
        'xǁSIRǁget_parameters_schema__mutmut_18': xǁSIRǁget_parameters_schema__mutmut_18, 
        'xǁSIRǁget_parameters_schema__mutmut_19': xǁSIRǁget_parameters_schema__mutmut_19, 
        'xǁSIRǁget_parameters_schema__mutmut_20': xǁSIRǁget_parameters_schema__mutmut_20, 
        'xǁSIRǁget_parameters_schema__mutmut_21': xǁSIRǁget_parameters_schema__mutmut_21, 
        'xǁSIRǁget_parameters_schema__mutmut_22': xǁSIRǁget_parameters_schema__mutmut_22, 
        'xǁSIRǁget_parameters_schema__mutmut_23': xǁSIRǁget_parameters_schema__mutmut_23, 
        'xǁSIRǁget_parameters_schema__mutmut_24': xǁSIRǁget_parameters_schema__mutmut_24, 
        'xǁSIRǁget_parameters_schema__mutmut_25': xǁSIRǁget_parameters_schema__mutmut_25, 
        'xǁSIRǁget_parameters_schema__mutmut_26': xǁSIRǁget_parameters_schema__mutmut_26, 
        'xǁSIRǁget_parameters_schema__mutmut_27': xǁSIRǁget_parameters_schema__mutmut_27, 
        'xǁSIRǁget_parameters_schema__mutmut_28': xǁSIRǁget_parameters_schema__mutmut_28, 
        'xǁSIRǁget_parameters_schema__mutmut_29': xǁSIRǁget_parameters_schema__mutmut_29, 
        'xǁSIRǁget_parameters_schema__mutmut_30': xǁSIRǁget_parameters_schema__mutmut_30, 
        'xǁSIRǁget_parameters_schema__mutmut_31': xǁSIRǁget_parameters_schema__mutmut_31, 
        'xǁSIRǁget_parameters_schema__mutmut_32': xǁSIRǁget_parameters_schema__mutmut_32, 
        'xǁSIRǁget_parameters_schema__mutmut_33': xǁSIRǁget_parameters_schema__mutmut_33, 
        'xǁSIRǁget_parameters_schema__mutmut_34': xǁSIRǁget_parameters_schema__mutmut_34, 
        'xǁSIRǁget_parameters_schema__mutmut_35': xǁSIRǁget_parameters_schema__mutmut_35, 
        'xǁSIRǁget_parameters_schema__mutmut_36': xǁSIRǁget_parameters_schema__mutmut_36, 
        'xǁSIRǁget_parameters_schema__mutmut_37': xǁSIRǁget_parameters_schema__mutmut_37, 
        'xǁSIRǁget_parameters_schema__mutmut_38': xǁSIRǁget_parameters_schema__mutmut_38, 
        'xǁSIRǁget_parameters_schema__mutmut_39': xǁSIRǁget_parameters_schema__mutmut_39, 
        'xǁSIRǁget_parameters_schema__mutmut_40': xǁSIRǁget_parameters_schema__mutmut_40, 
        'xǁSIRǁget_parameters_schema__mutmut_41': xǁSIRǁget_parameters_schema__mutmut_41, 
        'xǁSIRǁget_parameters_schema__mutmut_42': xǁSIRǁget_parameters_schema__mutmut_42, 
        'xǁSIRǁget_parameters_schema__mutmut_43': xǁSIRǁget_parameters_schema__mutmut_43, 
        'xǁSIRǁget_parameters_schema__mutmut_44': xǁSIRǁget_parameters_schema__mutmut_44, 
        'xǁSIRǁget_parameters_schema__mutmut_45': xǁSIRǁget_parameters_schema__mutmut_45, 
        'xǁSIRǁget_parameters_schema__mutmut_46': xǁSIRǁget_parameters_schema__mutmut_46, 
        'xǁSIRǁget_parameters_schema__mutmut_47': xǁSIRǁget_parameters_schema__mutmut_47, 
        'xǁSIRǁget_parameters_schema__mutmut_48': xǁSIRǁget_parameters_schema__mutmut_48, 
        'xǁSIRǁget_parameters_schema__mutmut_49': xǁSIRǁget_parameters_schema__mutmut_49, 
        'xǁSIRǁget_parameters_schema__mutmut_50': xǁSIRǁget_parameters_schema__mutmut_50, 
        'xǁSIRǁget_parameters_schema__mutmut_51': xǁSIRǁget_parameters_schema__mutmut_51, 
        'xǁSIRǁget_parameters_schema__mutmut_52': xǁSIRǁget_parameters_schema__mutmut_52, 
        'xǁSIRǁget_parameters_schema__mutmut_53': xǁSIRǁget_parameters_schema__mutmut_53, 
        'xǁSIRǁget_parameters_schema__mutmut_54': xǁSIRǁget_parameters_schema__mutmut_54, 
        'xǁSIRǁget_parameters_schema__mutmut_55': xǁSIRǁget_parameters_schema__mutmut_55, 
        'xǁSIRǁget_parameters_schema__mutmut_56': xǁSIRǁget_parameters_schema__mutmut_56, 
        'xǁSIRǁget_parameters_schema__mutmut_57': xǁSIRǁget_parameters_schema__mutmut_57, 
        'xǁSIRǁget_parameters_schema__mutmut_58': xǁSIRǁget_parameters_schema__mutmut_58, 
        'xǁSIRǁget_parameters_schema__mutmut_59': xǁSIRǁget_parameters_schema__mutmut_59, 
        'xǁSIRǁget_parameters_schema__mutmut_60': xǁSIRǁget_parameters_schema__mutmut_60, 
        'xǁSIRǁget_parameters_schema__mutmut_61': xǁSIRǁget_parameters_schema__mutmut_61, 
        'xǁSIRǁget_parameters_schema__mutmut_62': xǁSIRǁget_parameters_schema__mutmut_62, 
        'xǁSIRǁget_parameters_schema__mutmut_63': xǁSIRǁget_parameters_schema__mutmut_63, 
        'xǁSIRǁget_parameters_schema__mutmut_64': xǁSIRǁget_parameters_schema__mutmut_64, 
        'xǁSIRǁget_parameters_schema__mutmut_65': xǁSIRǁget_parameters_schema__mutmut_65, 
        'xǁSIRǁget_parameters_schema__mutmut_66': xǁSIRǁget_parameters_schema__mutmut_66, 
        'xǁSIRǁget_parameters_schema__mutmut_67': xǁSIRǁget_parameters_schema__mutmut_67, 
        'xǁSIRǁget_parameters_schema__mutmut_68': xǁSIRǁget_parameters_schema__mutmut_68
    }
    xǁSIRǁget_parameters_schema__mutmut_orig.__name__ = 'xǁSIRǁget_parameters_schema'


class SIS(ContagionSpread):
    """Implements the Susceptible-Infected-Susceptible (SIS) model."""

    def __init__(self, beta: float = 0.2, gamma: float = 0.1):
        args = [beta, gamma]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSISǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁSISǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁSISǁ__init____mutmut_orig(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = gamma

    def xǁSISǁ__init____mutmut_1(self, beta: float = 1.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = gamma

    def xǁSISǁ__init____mutmut_2(self, beta: float = 0.2, gamma: float = 1.1):
        self.beta = beta
        self.gamma = gamma

    def xǁSISǁ__init____mutmut_3(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = None
        self.gamma = gamma

    def xǁSISǁ__init____mutmut_4(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = None
    
    xǁSISǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSISǁ__init____mutmut_1': xǁSISǁ__init____mutmut_1, 
        'xǁSISǁ__init____mutmut_2': xǁSISǁ__init____mutmut_2, 
        'xǁSISǁ__init____mutmut_3': xǁSISǁ__init____mutmut_3, 
        'xǁSISǁ__init____mutmut_4': xǁSISǁ__init____mutmut_4
    }
    xǁSISǁ__init____mutmut_orig.__name__ = 'xǁSISǁ__init__'

    def differential(self, y: np.ndarray, t: float) -> np.ndarray:
        args = [y, t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSISǁdifferential__mutmut_orig'), object.__getattribute__(self, 'xǁSISǁdifferential__mutmut_mutants'), args, kwargs, self)

    def xǁSISǁdifferential__mutmut_orig(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I + self.gamma * I
        dIdt = self.beta * S * I - self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_1(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = None
        dSdt = -self.beta * S * I + self.gamma * I
        dIdt = self.beta * S * I - self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_2(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = None
        dIdt = self.beta * S * I - self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_3(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I - self.gamma * I
        dIdt = self.beta * S * I - self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_4(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S / I + self.gamma * I
        dIdt = self.beta * S * I - self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_5(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta / S * I + self.gamma * I
        dIdt = self.beta * S * I - self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_6(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = +self.beta * S * I + self.gamma * I
        dIdt = self.beta * S * I - self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_7(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I + self.gamma / I
        dIdt = self.beta * S * I - self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_8(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I + self.gamma * I
        dIdt = None
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_9(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I + self.gamma * I
        dIdt = self.beta * S * I + self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_10(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I + self.gamma * I
        dIdt = self.beta * S / I - self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_11(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I + self.gamma * I
        dIdt = self.beta / S * I - self.gamma * I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_12(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I + self.gamma * I
        dIdt = self.beta * S * I - self.gamma / I
        return np.array([dSdt, dIdt])

    def xǁSISǁdifferential__mutmut_13(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I + self.gamma * I
        dIdt = self.beta * S * I - self.gamma * I
        return np.array(None)
    
    xǁSISǁdifferential__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSISǁdifferential__mutmut_1': xǁSISǁdifferential__mutmut_1, 
        'xǁSISǁdifferential__mutmut_2': xǁSISǁdifferential__mutmut_2, 
        'xǁSISǁdifferential__mutmut_3': xǁSISǁdifferential__mutmut_3, 
        'xǁSISǁdifferential__mutmut_4': xǁSISǁdifferential__mutmut_4, 
        'xǁSISǁdifferential__mutmut_5': xǁSISǁdifferential__mutmut_5, 
        'xǁSISǁdifferential__mutmut_6': xǁSISǁdifferential__mutmut_6, 
        'xǁSISǁdifferential__mutmut_7': xǁSISǁdifferential__mutmut_7, 
        'xǁSISǁdifferential__mutmut_8': xǁSISǁdifferential__mutmut_8, 
        'xǁSISǁdifferential__mutmut_9': xǁSISǁdifferential__mutmut_9, 
        'xǁSISǁdifferential__mutmut_10': xǁSISǁdifferential__mutmut_10, 
        'xǁSISǁdifferential__mutmut_11': xǁSISǁdifferential__mutmut_11, 
        'xǁSISǁdifferential__mutmut_12': xǁSISǁdifferential__mutmut_12, 
        'xǁSISǁdifferential__mutmut_13': xǁSISǁdifferential__mutmut_13
    }
    xǁSISǁdifferential__mutmut_orig.__name__ = 'xǁSISǁdifferential'

    def compute_spread_rate(self, **params):
        args = []# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSISǁcompute_spread_rate__mutmut_orig'), object.__getattribute__(self, 'xǁSISǁcompute_spread_rate__mutmut_mutants'), args, kwargs, self)

    def xǁSISǁcompute_spread_rate__mutmut_orig(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_1(self, **params):
        """Calculates the instantaneous spread rate."""
        S = None
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_2(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get(None)
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_3(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("XXSXX")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_4(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("s")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_5(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = None
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_6(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get(None)
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_7(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("XXIXX")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_8(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("i")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_9(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = None
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_10(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get(None, self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_11(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", None)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_12(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get(self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_13(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", )
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_14(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("XXtransmission_rateXX", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_15(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("TRANSMISSION_RATE", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_16(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = None

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_17(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get(None, self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_18(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", None)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_19(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get(self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_20(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", )

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_21(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("XXrecovery_rateXX", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_22(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("RECOVERY_RATE", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_23(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = None
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_24(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I - gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_25(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S / I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_26(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta / S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_27(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = +beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_28(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma / I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_29(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = None
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_30(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I + gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_31(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S / I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_32(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta / S * I - gamma * I
        return dSdt, dIdt

    def xǁSISǁcompute_spread_rate__mutmut_33(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma / I
        return dSdt, dIdt
    
    xǁSISǁcompute_spread_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSISǁcompute_spread_rate__mutmut_1': xǁSISǁcompute_spread_rate__mutmut_1, 
        'xǁSISǁcompute_spread_rate__mutmut_2': xǁSISǁcompute_spread_rate__mutmut_2, 
        'xǁSISǁcompute_spread_rate__mutmut_3': xǁSISǁcompute_spread_rate__mutmut_3, 
        'xǁSISǁcompute_spread_rate__mutmut_4': xǁSISǁcompute_spread_rate__mutmut_4, 
        'xǁSISǁcompute_spread_rate__mutmut_5': xǁSISǁcompute_spread_rate__mutmut_5, 
        'xǁSISǁcompute_spread_rate__mutmut_6': xǁSISǁcompute_spread_rate__mutmut_6, 
        'xǁSISǁcompute_spread_rate__mutmut_7': xǁSISǁcompute_spread_rate__mutmut_7, 
        'xǁSISǁcompute_spread_rate__mutmut_8': xǁSISǁcompute_spread_rate__mutmut_8, 
        'xǁSISǁcompute_spread_rate__mutmut_9': xǁSISǁcompute_spread_rate__mutmut_9, 
        'xǁSISǁcompute_spread_rate__mutmut_10': xǁSISǁcompute_spread_rate__mutmut_10, 
        'xǁSISǁcompute_spread_rate__mutmut_11': xǁSISǁcompute_spread_rate__mutmut_11, 
        'xǁSISǁcompute_spread_rate__mutmut_12': xǁSISǁcompute_spread_rate__mutmut_12, 
        'xǁSISǁcompute_spread_rate__mutmut_13': xǁSISǁcompute_spread_rate__mutmut_13, 
        'xǁSISǁcompute_spread_rate__mutmut_14': xǁSISǁcompute_spread_rate__mutmut_14, 
        'xǁSISǁcompute_spread_rate__mutmut_15': xǁSISǁcompute_spread_rate__mutmut_15, 
        'xǁSISǁcompute_spread_rate__mutmut_16': xǁSISǁcompute_spread_rate__mutmut_16, 
        'xǁSISǁcompute_spread_rate__mutmut_17': xǁSISǁcompute_spread_rate__mutmut_17, 
        'xǁSISǁcompute_spread_rate__mutmut_18': xǁSISǁcompute_spread_rate__mutmut_18, 
        'xǁSISǁcompute_spread_rate__mutmut_19': xǁSISǁcompute_spread_rate__mutmut_19, 
        'xǁSISǁcompute_spread_rate__mutmut_20': xǁSISǁcompute_spread_rate__mutmut_20, 
        'xǁSISǁcompute_spread_rate__mutmut_21': xǁSISǁcompute_spread_rate__mutmut_21, 
        'xǁSISǁcompute_spread_rate__mutmut_22': xǁSISǁcompute_spread_rate__mutmut_22, 
        'xǁSISǁcompute_spread_rate__mutmut_23': xǁSISǁcompute_spread_rate__mutmut_23, 
        'xǁSISǁcompute_spread_rate__mutmut_24': xǁSISǁcompute_spread_rate__mutmut_24, 
        'xǁSISǁcompute_spread_rate__mutmut_25': xǁSISǁcompute_spread_rate__mutmut_25, 
        'xǁSISǁcompute_spread_rate__mutmut_26': xǁSISǁcompute_spread_rate__mutmut_26, 
        'xǁSISǁcompute_spread_rate__mutmut_27': xǁSISǁcompute_spread_rate__mutmut_27, 
        'xǁSISǁcompute_spread_rate__mutmut_28': xǁSISǁcompute_spread_rate__mutmut_28, 
        'xǁSISǁcompute_spread_rate__mutmut_29': xǁSISǁcompute_spread_rate__mutmut_29, 
        'xǁSISǁcompute_spread_rate__mutmut_30': xǁSISǁcompute_spread_rate__mutmut_30, 
        'xǁSISǁcompute_spread_rate__mutmut_31': xǁSISǁcompute_spread_rate__mutmut_31, 
        'xǁSISǁcompute_spread_rate__mutmut_32': xǁSISǁcompute_spread_rate__mutmut_32, 
        'xǁSISǁcompute_spread_rate__mutmut_33': xǁSISǁcompute_spread_rate__mutmut_33
    }
    xǁSISǁcompute_spread_rate__mutmut_orig.__name__ = 'xǁSISǁcompute_spread_rate'

    def predict_states(self, time_points, **params):
        args = [time_points]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSISǁpredict_states__mutmut_orig'), object.__getattribute__(self, 'xǁSISǁpredict_states__mutmut_mutants'), args, kwargs, self)

    def xǁSISǁpredict_states__mutmut_orig(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_1(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = None
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_2(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get(None, 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_3(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", None)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_4(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get(999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_5(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", )
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_6(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("XXS0XX", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_7(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("s0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_8(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 1000)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_9(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = None

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_10(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get(None, 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_11(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", None)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_12(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get(1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_13(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", )

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_14(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("XXI0XX", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_15(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("i0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_16(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 2)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_17(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=None, I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_18(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=None, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_19(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_20(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_21(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], )

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_22(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[1], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_23(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_24(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = None
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_25(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            None,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_26(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            None,
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_27(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            None,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_28(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=None,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_29(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method=None,
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_30(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_31(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_32(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_33(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_34(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_35(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[1], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_36(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[+1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_37(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-2]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_38(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="XXLSODAXX",
        )
        return sol.y.T

    def xǁSISǁpredict_states__mutmut_39(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="lsoda",
        )
        return sol.y.T
    
    xǁSISǁpredict_states__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSISǁpredict_states__mutmut_1': xǁSISǁpredict_states__mutmut_1, 
        'xǁSISǁpredict_states__mutmut_2': xǁSISǁpredict_states__mutmut_2, 
        'xǁSISǁpredict_states__mutmut_3': xǁSISǁpredict_states__mutmut_3, 
        'xǁSISǁpredict_states__mutmut_4': xǁSISǁpredict_states__mutmut_4, 
        'xǁSISǁpredict_states__mutmut_5': xǁSISǁpredict_states__mutmut_5, 
        'xǁSISǁpredict_states__mutmut_6': xǁSISǁpredict_states__mutmut_6, 
        'xǁSISǁpredict_states__mutmut_7': xǁSISǁpredict_states__mutmut_7, 
        'xǁSISǁpredict_states__mutmut_8': xǁSISǁpredict_states__mutmut_8, 
        'xǁSISǁpredict_states__mutmut_9': xǁSISǁpredict_states__mutmut_9, 
        'xǁSISǁpredict_states__mutmut_10': xǁSISǁpredict_states__mutmut_10, 
        'xǁSISǁpredict_states__mutmut_11': xǁSISǁpredict_states__mutmut_11, 
        'xǁSISǁpredict_states__mutmut_12': xǁSISǁpredict_states__mutmut_12, 
        'xǁSISǁpredict_states__mutmut_13': xǁSISǁpredict_states__mutmut_13, 
        'xǁSISǁpredict_states__mutmut_14': xǁSISǁpredict_states__mutmut_14, 
        'xǁSISǁpredict_states__mutmut_15': xǁSISǁpredict_states__mutmut_15, 
        'xǁSISǁpredict_states__mutmut_16': xǁSISǁpredict_states__mutmut_16, 
        'xǁSISǁpredict_states__mutmut_17': xǁSISǁpredict_states__mutmut_17, 
        'xǁSISǁpredict_states__mutmut_18': xǁSISǁpredict_states__mutmut_18, 
        'xǁSISǁpredict_states__mutmut_19': xǁSISǁpredict_states__mutmut_19, 
        'xǁSISǁpredict_states__mutmut_20': xǁSISǁpredict_states__mutmut_20, 
        'xǁSISǁpredict_states__mutmut_21': xǁSISǁpredict_states__mutmut_21, 
        'xǁSISǁpredict_states__mutmut_22': xǁSISǁpredict_states__mutmut_22, 
        'xǁSISǁpredict_states__mutmut_23': xǁSISǁpredict_states__mutmut_23, 
        'xǁSISǁpredict_states__mutmut_24': xǁSISǁpredict_states__mutmut_24, 
        'xǁSISǁpredict_states__mutmut_25': xǁSISǁpredict_states__mutmut_25, 
        'xǁSISǁpredict_states__mutmut_26': xǁSISǁpredict_states__mutmut_26, 
        'xǁSISǁpredict_states__mutmut_27': xǁSISǁpredict_states__mutmut_27, 
        'xǁSISǁpredict_states__mutmut_28': xǁSISǁpredict_states__mutmut_28, 
        'xǁSISǁpredict_states__mutmut_29': xǁSISǁpredict_states__mutmut_29, 
        'xǁSISǁpredict_states__mutmut_30': xǁSISǁpredict_states__mutmut_30, 
        'xǁSISǁpredict_states__mutmut_31': xǁSISǁpredict_states__mutmut_31, 
        'xǁSISǁpredict_states__mutmut_32': xǁSISǁpredict_states__mutmut_32, 
        'xǁSISǁpredict_states__mutmut_33': xǁSISǁpredict_states__mutmut_33, 
        'xǁSISǁpredict_states__mutmut_34': xǁSISǁpredict_states__mutmut_34, 
        'xǁSISǁpredict_states__mutmut_35': xǁSISǁpredict_states__mutmut_35, 
        'xǁSISǁpredict_states__mutmut_36': xǁSISǁpredict_states__mutmut_36, 
        'xǁSISǁpredict_states__mutmut_37': xǁSISǁpredict_states__mutmut_37, 
        'xǁSISǁpredict_states__mutmut_38': xǁSISǁpredict_states__mutmut_38, 
        'xǁSISǁpredict_states__mutmut_39': xǁSISǁpredict_states__mutmut_39
    }
    xǁSISǁpredict_states__mutmut_orig.__name__ = 'xǁSISǁpredict_states'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSISǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁSISǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁSISǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters."""
        return {
            "XXtransmission_rateXX": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters."""
        return {
            "TRANSMISSION_RATE": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "XXtypeXX": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "TYPE": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "XXfloatXX",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "FLOAT",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "XXdefaultXX": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "DEFAULT": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "XXdescriptionXX": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "DESCRIPTION": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "XXThe rate of transmission of the contagion.XX",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "the rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "THE RATE OF TRANSMISSION OF THE CONTAGION.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_14(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "XXrecovery_rateXX": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_15(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "RECOVERY_RATE": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_16(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "XXtypeXX": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_17(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "TYPE": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_18(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "XXfloatXX",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_19(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "FLOAT",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_20(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "XXdefaultXX": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_21(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "DEFAULT": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_22(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "XXdescriptionXX": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_23(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "DESCRIPTION": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_24(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "XXThe rate at which infectious individuals return to susceptible state.XX",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_25(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "the rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_26(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "THE RATE AT WHICH INFECTIOUS INDIVIDUALS RETURN TO SUSCEPTIBLE STATE.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_27(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "XXS0XX": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_28(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "s0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_29(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "XXtypeXX": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_30(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "TYPE": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_31(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "XXfloatXX",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_32(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "FLOAT",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_33(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "XXdefaultXX": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_34(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "DEFAULT": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_35(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 1000,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_36(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "XXdescriptionXX": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_37(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "DESCRIPTION": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_38(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "XXThe initial number of susceptible individuals.XX",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_39(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "the initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_40(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "THE INITIAL NUMBER OF SUSCEPTIBLE INDIVIDUALS.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_41(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "XXI0XX": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_42(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "i0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_43(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "XXtypeXX": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_44(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "TYPE": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_45(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "XXfloatXX",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_46(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "FLOAT",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_47(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "XXdefaultXX": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_48(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "DEFAULT": 1,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_49(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 2,
                "description": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_50(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "XXdescriptionXX": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_51(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "DESCRIPTION": "The initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_52(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "XXThe initial number of infectious individuals.XX",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_53(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "the initial number of infectious individuals.",
            },
        }

    def xǁSISǁget_parameters_schema__mutmut_54(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "THE INITIAL NUMBER OF INFECTIOUS INDIVIDUALS.",
            },
        }
    
    xǁSISǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSISǁget_parameters_schema__mutmut_1': xǁSISǁget_parameters_schema__mutmut_1, 
        'xǁSISǁget_parameters_schema__mutmut_2': xǁSISǁget_parameters_schema__mutmut_2, 
        'xǁSISǁget_parameters_schema__mutmut_3': xǁSISǁget_parameters_schema__mutmut_3, 
        'xǁSISǁget_parameters_schema__mutmut_4': xǁSISǁget_parameters_schema__mutmut_4, 
        'xǁSISǁget_parameters_schema__mutmut_5': xǁSISǁget_parameters_schema__mutmut_5, 
        'xǁSISǁget_parameters_schema__mutmut_6': xǁSISǁget_parameters_schema__mutmut_6, 
        'xǁSISǁget_parameters_schema__mutmut_7': xǁSISǁget_parameters_schema__mutmut_7, 
        'xǁSISǁget_parameters_schema__mutmut_8': xǁSISǁget_parameters_schema__mutmut_8, 
        'xǁSISǁget_parameters_schema__mutmut_9': xǁSISǁget_parameters_schema__mutmut_9, 
        'xǁSISǁget_parameters_schema__mutmut_10': xǁSISǁget_parameters_schema__mutmut_10, 
        'xǁSISǁget_parameters_schema__mutmut_11': xǁSISǁget_parameters_schema__mutmut_11, 
        'xǁSISǁget_parameters_schema__mutmut_12': xǁSISǁget_parameters_schema__mutmut_12, 
        'xǁSISǁget_parameters_schema__mutmut_13': xǁSISǁget_parameters_schema__mutmut_13, 
        'xǁSISǁget_parameters_schema__mutmut_14': xǁSISǁget_parameters_schema__mutmut_14, 
        'xǁSISǁget_parameters_schema__mutmut_15': xǁSISǁget_parameters_schema__mutmut_15, 
        'xǁSISǁget_parameters_schema__mutmut_16': xǁSISǁget_parameters_schema__mutmut_16, 
        'xǁSISǁget_parameters_schema__mutmut_17': xǁSISǁget_parameters_schema__mutmut_17, 
        'xǁSISǁget_parameters_schema__mutmut_18': xǁSISǁget_parameters_schema__mutmut_18, 
        'xǁSISǁget_parameters_schema__mutmut_19': xǁSISǁget_parameters_schema__mutmut_19, 
        'xǁSISǁget_parameters_schema__mutmut_20': xǁSISǁget_parameters_schema__mutmut_20, 
        'xǁSISǁget_parameters_schema__mutmut_21': xǁSISǁget_parameters_schema__mutmut_21, 
        'xǁSISǁget_parameters_schema__mutmut_22': xǁSISǁget_parameters_schema__mutmut_22, 
        'xǁSISǁget_parameters_schema__mutmut_23': xǁSISǁget_parameters_schema__mutmut_23, 
        'xǁSISǁget_parameters_schema__mutmut_24': xǁSISǁget_parameters_schema__mutmut_24, 
        'xǁSISǁget_parameters_schema__mutmut_25': xǁSISǁget_parameters_schema__mutmut_25, 
        'xǁSISǁget_parameters_schema__mutmut_26': xǁSISǁget_parameters_schema__mutmut_26, 
        'xǁSISǁget_parameters_schema__mutmut_27': xǁSISǁget_parameters_schema__mutmut_27, 
        'xǁSISǁget_parameters_schema__mutmut_28': xǁSISǁget_parameters_schema__mutmut_28, 
        'xǁSISǁget_parameters_schema__mutmut_29': xǁSISǁget_parameters_schema__mutmut_29, 
        'xǁSISǁget_parameters_schema__mutmut_30': xǁSISǁget_parameters_schema__mutmut_30, 
        'xǁSISǁget_parameters_schema__mutmut_31': xǁSISǁget_parameters_schema__mutmut_31, 
        'xǁSISǁget_parameters_schema__mutmut_32': xǁSISǁget_parameters_schema__mutmut_32, 
        'xǁSISǁget_parameters_schema__mutmut_33': xǁSISǁget_parameters_schema__mutmut_33, 
        'xǁSISǁget_parameters_schema__mutmut_34': xǁSISǁget_parameters_schema__mutmut_34, 
        'xǁSISǁget_parameters_schema__mutmut_35': xǁSISǁget_parameters_schema__mutmut_35, 
        'xǁSISǁget_parameters_schema__mutmut_36': xǁSISǁget_parameters_schema__mutmut_36, 
        'xǁSISǁget_parameters_schema__mutmut_37': xǁSISǁget_parameters_schema__mutmut_37, 
        'xǁSISǁget_parameters_schema__mutmut_38': xǁSISǁget_parameters_schema__mutmut_38, 
        'xǁSISǁget_parameters_schema__mutmut_39': xǁSISǁget_parameters_schema__mutmut_39, 
        'xǁSISǁget_parameters_schema__mutmut_40': xǁSISǁget_parameters_schema__mutmut_40, 
        'xǁSISǁget_parameters_schema__mutmut_41': xǁSISǁget_parameters_schema__mutmut_41, 
        'xǁSISǁget_parameters_schema__mutmut_42': xǁSISǁget_parameters_schema__mutmut_42, 
        'xǁSISǁget_parameters_schema__mutmut_43': xǁSISǁget_parameters_schema__mutmut_43, 
        'xǁSISǁget_parameters_schema__mutmut_44': xǁSISǁget_parameters_schema__mutmut_44, 
        'xǁSISǁget_parameters_schema__mutmut_45': xǁSISǁget_parameters_schema__mutmut_45, 
        'xǁSISǁget_parameters_schema__mutmut_46': xǁSISǁget_parameters_schema__mutmut_46, 
        'xǁSISǁget_parameters_schema__mutmut_47': xǁSISǁget_parameters_schema__mutmut_47, 
        'xǁSISǁget_parameters_schema__mutmut_48': xǁSISǁget_parameters_schema__mutmut_48, 
        'xǁSISǁget_parameters_schema__mutmut_49': xǁSISǁget_parameters_schema__mutmut_49, 
        'xǁSISǁget_parameters_schema__mutmut_50': xǁSISǁget_parameters_schema__mutmut_50, 
        'xǁSISǁget_parameters_schema__mutmut_51': xǁSISǁget_parameters_schema__mutmut_51, 
        'xǁSISǁget_parameters_schema__mutmut_52': xǁSISǁget_parameters_schema__mutmut_52, 
        'xǁSISǁget_parameters_schema__mutmut_53': xǁSISǁget_parameters_schema__mutmut_53, 
        'xǁSISǁget_parameters_schema__mutmut_54': xǁSISǁget_parameters_schema__mutmut_54
    }
    xǁSISǁget_parameters_schema__mutmut_orig.__name__ = 'xǁSISǁget_parameters_schema'


class SEIR(ContagionSpread):
    """Implements the Susceptible-Exposed-Infected-Recovered (SEIR) model."""

    def __init__(self, beta: float = 0.2, sigma: float = 0.5, gamma: float = 0.1):
        args = [beta, sigma, gamma]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSEIRǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁSEIRǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁSEIRǁ__init____mutmut_orig(self, beta: float = 0.2, sigma: float = 0.5, gamma: float = 0.1):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

    def xǁSEIRǁ__init____mutmut_1(self, beta: float = 1.2, sigma: float = 0.5, gamma: float = 0.1):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

    def xǁSEIRǁ__init____mutmut_2(self, beta: float = 0.2, sigma: float = 1.5, gamma: float = 0.1):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

    def xǁSEIRǁ__init____mutmut_3(self, beta: float = 0.2, sigma: float = 0.5, gamma: float = 1.1):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

    def xǁSEIRǁ__init____mutmut_4(self, beta: float = 0.2, sigma: float = 0.5, gamma: float = 0.1):
        self.beta = None
        self.sigma = sigma
        self.gamma = gamma

    def xǁSEIRǁ__init____mutmut_5(self, beta: float = 0.2, sigma: float = 0.5, gamma: float = 0.1):
        self.beta = beta
        self.sigma = None
        self.gamma = gamma

    def xǁSEIRǁ__init____mutmut_6(self, beta: float = 0.2, sigma: float = 0.5, gamma: float = 0.1):
        self.beta = beta
        self.sigma = sigma
        self.gamma = None
    
    xǁSEIRǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSEIRǁ__init____mutmut_1': xǁSEIRǁ__init____mutmut_1, 
        'xǁSEIRǁ__init____mutmut_2': xǁSEIRǁ__init____mutmut_2, 
        'xǁSEIRǁ__init____mutmut_3': xǁSEIRǁ__init____mutmut_3, 
        'xǁSEIRǁ__init____mutmut_4': xǁSEIRǁ__init____mutmut_4, 
        'xǁSEIRǁ__init____mutmut_5': xǁSEIRǁ__init____mutmut_5, 
        'xǁSEIRǁ__init____mutmut_6': xǁSEIRǁ__init____mutmut_6
    }
    xǁSEIRǁ__init____mutmut_orig.__name__ = 'xǁSEIRǁ__init__'

    def differential(self, y: np.ndarray, t: float) -> np.ndarray:
        args = [y, t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSEIRǁdifferential__mutmut_orig'), object.__getattribute__(self, 'xǁSEIRǁdifferential__mutmut_mutants'), args, kwargs, self)

    def xǁSEIRǁdifferential__mutmut_orig(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_1(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = None
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_2(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = None
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_3(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S / I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_4(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta / S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_5(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = +self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_6(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = None
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_7(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I + self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_8(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S / I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_9(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta / S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_10(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma / E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_11(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = None
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_12(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E + self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_13(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma / E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_14(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma / I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_15(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = None
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_16(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma / I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def xǁSEIRǁdifferential__mutmut_17(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array(None)
    
    xǁSEIRǁdifferential__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSEIRǁdifferential__mutmut_1': xǁSEIRǁdifferential__mutmut_1, 
        'xǁSEIRǁdifferential__mutmut_2': xǁSEIRǁdifferential__mutmut_2, 
        'xǁSEIRǁdifferential__mutmut_3': xǁSEIRǁdifferential__mutmut_3, 
        'xǁSEIRǁdifferential__mutmut_4': xǁSEIRǁdifferential__mutmut_4, 
        'xǁSEIRǁdifferential__mutmut_5': xǁSEIRǁdifferential__mutmut_5, 
        'xǁSEIRǁdifferential__mutmut_6': xǁSEIRǁdifferential__mutmut_6, 
        'xǁSEIRǁdifferential__mutmut_7': xǁSEIRǁdifferential__mutmut_7, 
        'xǁSEIRǁdifferential__mutmut_8': xǁSEIRǁdifferential__mutmut_8, 
        'xǁSEIRǁdifferential__mutmut_9': xǁSEIRǁdifferential__mutmut_9, 
        'xǁSEIRǁdifferential__mutmut_10': xǁSEIRǁdifferential__mutmut_10, 
        'xǁSEIRǁdifferential__mutmut_11': xǁSEIRǁdifferential__mutmut_11, 
        'xǁSEIRǁdifferential__mutmut_12': xǁSEIRǁdifferential__mutmut_12, 
        'xǁSEIRǁdifferential__mutmut_13': xǁSEIRǁdifferential__mutmut_13, 
        'xǁSEIRǁdifferential__mutmut_14': xǁSEIRǁdifferential__mutmut_14, 
        'xǁSEIRǁdifferential__mutmut_15': xǁSEIRǁdifferential__mutmut_15, 
        'xǁSEIRǁdifferential__mutmut_16': xǁSEIRǁdifferential__mutmut_16, 
        'xǁSEIRǁdifferential__mutmut_17': xǁSEIRǁdifferential__mutmut_17
    }
    xǁSEIRǁdifferential__mutmut_orig.__name__ = 'xǁSEIRǁdifferential'

    def compute_spread_rate(self, **params):
        args = []# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSEIRǁcompute_spread_rate__mutmut_orig'), object.__getattribute__(self, 'xǁSEIRǁcompute_spread_rate__mutmut_mutants'), args, kwargs, self)

    def xǁSEIRǁcompute_spread_rate__mutmut_orig(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_1(self, **params):
        """Calculates the instantaneous spread rate."""
        S = None
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_2(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get(None)
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_3(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("XXSXX")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_4(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("s")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_5(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = None
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_6(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get(None)
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_7(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("XXEXX")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_8(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("e")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_9(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = None
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_10(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get(None)
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_11(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("XXIXX")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_12(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("i")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_13(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = None
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_14(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get(None, self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_15(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", None)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_16(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get(self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_17(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", )
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_18(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("XXtransmission_rateXX", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_19(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("TRANSMISSION_RATE", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_20(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = None
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_21(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get(None, self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_22(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", None)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_23(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get(self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_24(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", )
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_25(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("XXincubation_rateXX", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_26(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("INCUBATION_RATE", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_27(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = None

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_28(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get(None, self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_29(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", None)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_30(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get(self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_31(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", )

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_32(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("XXrecovery_rateXX", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_33(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("RECOVERY_RATE", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_34(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = None
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_35(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S / I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_36(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta / S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_37(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = +beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_38(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = None
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_39(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I + sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_40(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S / I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_41(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta / S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_42(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma / E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_43(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = None
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_44(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E + gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_45(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma / E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_46(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma / I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_47(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = None
        return dSdt, dEdt, dIdt, dRdt

    def xǁSEIRǁcompute_spread_rate__mutmut_48(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma / I
        return dSdt, dEdt, dIdt, dRdt
    
    xǁSEIRǁcompute_spread_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSEIRǁcompute_spread_rate__mutmut_1': xǁSEIRǁcompute_spread_rate__mutmut_1, 
        'xǁSEIRǁcompute_spread_rate__mutmut_2': xǁSEIRǁcompute_spread_rate__mutmut_2, 
        'xǁSEIRǁcompute_spread_rate__mutmut_3': xǁSEIRǁcompute_spread_rate__mutmut_3, 
        'xǁSEIRǁcompute_spread_rate__mutmut_4': xǁSEIRǁcompute_spread_rate__mutmut_4, 
        'xǁSEIRǁcompute_spread_rate__mutmut_5': xǁSEIRǁcompute_spread_rate__mutmut_5, 
        'xǁSEIRǁcompute_spread_rate__mutmut_6': xǁSEIRǁcompute_spread_rate__mutmut_6, 
        'xǁSEIRǁcompute_spread_rate__mutmut_7': xǁSEIRǁcompute_spread_rate__mutmut_7, 
        'xǁSEIRǁcompute_spread_rate__mutmut_8': xǁSEIRǁcompute_spread_rate__mutmut_8, 
        'xǁSEIRǁcompute_spread_rate__mutmut_9': xǁSEIRǁcompute_spread_rate__mutmut_9, 
        'xǁSEIRǁcompute_spread_rate__mutmut_10': xǁSEIRǁcompute_spread_rate__mutmut_10, 
        'xǁSEIRǁcompute_spread_rate__mutmut_11': xǁSEIRǁcompute_spread_rate__mutmut_11, 
        'xǁSEIRǁcompute_spread_rate__mutmut_12': xǁSEIRǁcompute_spread_rate__mutmut_12, 
        'xǁSEIRǁcompute_spread_rate__mutmut_13': xǁSEIRǁcompute_spread_rate__mutmut_13, 
        'xǁSEIRǁcompute_spread_rate__mutmut_14': xǁSEIRǁcompute_spread_rate__mutmut_14, 
        'xǁSEIRǁcompute_spread_rate__mutmut_15': xǁSEIRǁcompute_spread_rate__mutmut_15, 
        'xǁSEIRǁcompute_spread_rate__mutmut_16': xǁSEIRǁcompute_spread_rate__mutmut_16, 
        'xǁSEIRǁcompute_spread_rate__mutmut_17': xǁSEIRǁcompute_spread_rate__mutmut_17, 
        'xǁSEIRǁcompute_spread_rate__mutmut_18': xǁSEIRǁcompute_spread_rate__mutmut_18, 
        'xǁSEIRǁcompute_spread_rate__mutmut_19': xǁSEIRǁcompute_spread_rate__mutmut_19, 
        'xǁSEIRǁcompute_spread_rate__mutmut_20': xǁSEIRǁcompute_spread_rate__mutmut_20, 
        'xǁSEIRǁcompute_spread_rate__mutmut_21': xǁSEIRǁcompute_spread_rate__mutmut_21, 
        'xǁSEIRǁcompute_spread_rate__mutmut_22': xǁSEIRǁcompute_spread_rate__mutmut_22, 
        'xǁSEIRǁcompute_spread_rate__mutmut_23': xǁSEIRǁcompute_spread_rate__mutmut_23, 
        'xǁSEIRǁcompute_spread_rate__mutmut_24': xǁSEIRǁcompute_spread_rate__mutmut_24, 
        'xǁSEIRǁcompute_spread_rate__mutmut_25': xǁSEIRǁcompute_spread_rate__mutmut_25, 
        'xǁSEIRǁcompute_spread_rate__mutmut_26': xǁSEIRǁcompute_spread_rate__mutmut_26, 
        'xǁSEIRǁcompute_spread_rate__mutmut_27': xǁSEIRǁcompute_spread_rate__mutmut_27, 
        'xǁSEIRǁcompute_spread_rate__mutmut_28': xǁSEIRǁcompute_spread_rate__mutmut_28, 
        'xǁSEIRǁcompute_spread_rate__mutmut_29': xǁSEIRǁcompute_spread_rate__mutmut_29, 
        'xǁSEIRǁcompute_spread_rate__mutmut_30': xǁSEIRǁcompute_spread_rate__mutmut_30, 
        'xǁSEIRǁcompute_spread_rate__mutmut_31': xǁSEIRǁcompute_spread_rate__mutmut_31, 
        'xǁSEIRǁcompute_spread_rate__mutmut_32': xǁSEIRǁcompute_spread_rate__mutmut_32, 
        'xǁSEIRǁcompute_spread_rate__mutmut_33': xǁSEIRǁcompute_spread_rate__mutmut_33, 
        'xǁSEIRǁcompute_spread_rate__mutmut_34': xǁSEIRǁcompute_spread_rate__mutmut_34, 
        'xǁSEIRǁcompute_spread_rate__mutmut_35': xǁSEIRǁcompute_spread_rate__mutmut_35, 
        'xǁSEIRǁcompute_spread_rate__mutmut_36': xǁSEIRǁcompute_spread_rate__mutmut_36, 
        'xǁSEIRǁcompute_spread_rate__mutmut_37': xǁSEIRǁcompute_spread_rate__mutmut_37, 
        'xǁSEIRǁcompute_spread_rate__mutmut_38': xǁSEIRǁcompute_spread_rate__mutmut_38, 
        'xǁSEIRǁcompute_spread_rate__mutmut_39': xǁSEIRǁcompute_spread_rate__mutmut_39, 
        'xǁSEIRǁcompute_spread_rate__mutmut_40': xǁSEIRǁcompute_spread_rate__mutmut_40, 
        'xǁSEIRǁcompute_spread_rate__mutmut_41': xǁSEIRǁcompute_spread_rate__mutmut_41, 
        'xǁSEIRǁcompute_spread_rate__mutmut_42': xǁSEIRǁcompute_spread_rate__mutmut_42, 
        'xǁSEIRǁcompute_spread_rate__mutmut_43': xǁSEIRǁcompute_spread_rate__mutmut_43, 
        'xǁSEIRǁcompute_spread_rate__mutmut_44': xǁSEIRǁcompute_spread_rate__mutmut_44, 
        'xǁSEIRǁcompute_spread_rate__mutmut_45': xǁSEIRǁcompute_spread_rate__mutmut_45, 
        'xǁSEIRǁcompute_spread_rate__mutmut_46': xǁSEIRǁcompute_spread_rate__mutmut_46, 
        'xǁSEIRǁcompute_spread_rate__mutmut_47': xǁSEIRǁcompute_spread_rate__mutmut_47, 
        'xǁSEIRǁcompute_spread_rate__mutmut_48': xǁSEIRǁcompute_spread_rate__mutmut_48
    }
    xǁSEIRǁcompute_spread_rate__mutmut_orig.__name__ = 'xǁSEIRǁcompute_spread_rate'

    def predict_states(self, time_points, **params):
        args = [time_points]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSEIRǁpredict_states__mutmut_orig'), object.__getattribute__(self, 'xǁSEIRǁpredict_states__mutmut_mutants'), args, kwargs, self)

    def xǁSEIRǁpredict_states__mutmut_orig(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_1(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = None
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_2(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get(None, 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_3(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", None)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_4(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get(999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_5(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", )
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_6(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("XXS0XX", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_7(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("s0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_8(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 1000)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_9(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = None
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_10(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get(None, 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_11(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", None)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_12(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get(0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_13(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", )
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_14(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("XXE0XX", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_15(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("e0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_16(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 1)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_17(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = None
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_18(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get(None, 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_19(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", None)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_20(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get(1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_21(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", )
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_22(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("XXI0XX", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_23(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("i0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_24(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 2)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_25(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = None

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_26(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get(None, 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_27(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", None)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_28(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get(0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_29(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", )

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_30(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("XXR0XX", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_31(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("r0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_32(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_33(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=None, E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_34(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=None, I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_35(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=None, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_36(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_37(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_38(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_39(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], )

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_40(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[1], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_41(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[2], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_42(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[3], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_43(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = None
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_44(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            None,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_45(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            None,
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_46(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            None,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_47(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=None,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_48(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method=None,
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_49(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_50(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_51(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_52(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_53(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_54(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[1], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_55(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[+1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_56(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-2]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_57(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="XXLSODAXX",
        )
        return sol.y.T

    def xǁSEIRǁpredict_states__mutmut_58(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="lsoda",
        )
        return sol.y.T
    
    xǁSEIRǁpredict_states__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSEIRǁpredict_states__mutmut_1': xǁSEIRǁpredict_states__mutmut_1, 
        'xǁSEIRǁpredict_states__mutmut_2': xǁSEIRǁpredict_states__mutmut_2, 
        'xǁSEIRǁpredict_states__mutmut_3': xǁSEIRǁpredict_states__mutmut_3, 
        'xǁSEIRǁpredict_states__mutmut_4': xǁSEIRǁpredict_states__mutmut_4, 
        'xǁSEIRǁpredict_states__mutmut_5': xǁSEIRǁpredict_states__mutmut_5, 
        'xǁSEIRǁpredict_states__mutmut_6': xǁSEIRǁpredict_states__mutmut_6, 
        'xǁSEIRǁpredict_states__mutmut_7': xǁSEIRǁpredict_states__mutmut_7, 
        'xǁSEIRǁpredict_states__mutmut_8': xǁSEIRǁpredict_states__mutmut_8, 
        'xǁSEIRǁpredict_states__mutmut_9': xǁSEIRǁpredict_states__mutmut_9, 
        'xǁSEIRǁpredict_states__mutmut_10': xǁSEIRǁpredict_states__mutmut_10, 
        'xǁSEIRǁpredict_states__mutmut_11': xǁSEIRǁpredict_states__mutmut_11, 
        'xǁSEIRǁpredict_states__mutmut_12': xǁSEIRǁpredict_states__mutmut_12, 
        'xǁSEIRǁpredict_states__mutmut_13': xǁSEIRǁpredict_states__mutmut_13, 
        'xǁSEIRǁpredict_states__mutmut_14': xǁSEIRǁpredict_states__mutmut_14, 
        'xǁSEIRǁpredict_states__mutmut_15': xǁSEIRǁpredict_states__mutmut_15, 
        'xǁSEIRǁpredict_states__mutmut_16': xǁSEIRǁpredict_states__mutmut_16, 
        'xǁSEIRǁpredict_states__mutmut_17': xǁSEIRǁpredict_states__mutmut_17, 
        'xǁSEIRǁpredict_states__mutmut_18': xǁSEIRǁpredict_states__mutmut_18, 
        'xǁSEIRǁpredict_states__mutmut_19': xǁSEIRǁpredict_states__mutmut_19, 
        'xǁSEIRǁpredict_states__mutmut_20': xǁSEIRǁpredict_states__mutmut_20, 
        'xǁSEIRǁpredict_states__mutmut_21': xǁSEIRǁpredict_states__mutmut_21, 
        'xǁSEIRǁpredict_states__mutmut_22': xǁSEIRǁpredict_states__mutmut_22, 
        'xǁSEIRǁpredict_states__mutmut_23': xǁSEIRǁpredict_states__mutmut_23, 
        'xǁSEIRǁpredict_states__mutmut_24': xǁSEIRǁpredict_states__mutmut_24, 
        'xǁSEIRǁpredict_states__mutmut_25': xǁSEIRǁpredict_states__mutmut_25, 
        'xǁSEIRǁpredict_states__mutmut_26': xǁSEIRǁpredict_states__mutmut_26, 
        'xǁSEIRǁpredict_states__mutmut_27': xǁSEIRǁpredict_states__mutmut_27, 
        'xǁSEIRǁpredict_states__mutmut_28': xǁSEIRǁpredict_states__mutmut_28, 
        'xǁSEIRǁpredict_states__mutmut_29': xǁSEIRǁpredict_states__mutmut_29, 
        'xǁSEIRǁpredict_states__mutmut_30': xǁSEIRǁpredict_states__mutmut_30, 
        'xǁSEIRǁpredict_states__mutmut_31': xǁSEIRǁpredict_states__mutmut_31, 
        'xǁSEIRǁpredict_states__mutmut_32': xǁSEIRǁpredict_states__mutmut_32, 
        'xǁSEIRǁpredict_states__mutmut_33': xǁSEIRǁpredict_states__mutmut_33, 
        'xǁSEIRǁpredict_states__mutmut_34': xǁSEIRǁpredict_states__mutmut_34, 
        'xǁSEIRǁpredict_states__mutmut_35': xǁSEIRǁpredict_states__mutmut_35, 
        'xǁSEIRǁpredict_states__mutmut_36': xǁSEIRǁpredict_states__mutmut_36, 
        'xǁSEIRǁpredict_states__mutmut_37': xǁSEIRǁpredict_states__mutmut_37, 
        'xǁSEIRǁpredict_states__mutmut_38': xǁSEIRǁpredict_states__mutmut_38, 
        'xǁSEIRǁpredict_states__mutmut_39': xǁSEIRǁpredict_states__mutmut_39, 
        'xǁSEIRǁpredict_states__mutmut_40': xǁSEIRǁpredict_states__mutmut_40, 
        'xǁSEIRǁpredict_states__mutmut_41': xǁSEIRǁpredict_states__mutmut_41, 
        'xǁSEIRǁpredict_states__mutmut_42': xǁSEIRǁpredict_states__mutmut_42, 
        'xǁSEIRǁpredict_states__mutmut_43': xǁSEIRǁpredict_states__mutmut_43, 
        'xǁSEIRǁpredict_states__mutmut_44': xǁSEIRǁpredict_states__mutmut_44, 
        'xǁSEIRǁpredict_states__mutmut_45': xǁSEIRǁpredict_states__mutmut_45, 
        'xǁSEIRǁpredict_states__mutmut_46': xǁSEIRǁpredict_states__mutmut_46, 
        'xǁSEIRǁpredict_states__mutmut_47': xǁSEIRǁpredict_states__mutmut_47, 
        'xǁSEIRǁpredict_states__mutmut_48': xǁSEIRǁpredict_states__mutmut_48, 
        'xǁSEIRǁpredict_states__mutmut_49': xǁSEIRǁpredict_states__mutmut_49, 
        'xǁSEIRǁpredict_states__mutmut_50': xǁSEIRǁpredict_states__mutmut_50, 
        'xǁSEIRǁpredict_states__mutmut_51': xǁSEIRǁpredict_states__mutmut_51, 
        'xǁSEIRǁpredict_states__mutmut_52': xǁSEIRǁpredict_states__mutmut_52, 
        'xǁSEIRǁpredict_states__mutmut_53': xǁSEIRǁpredict_states__mutmut_53, 
        'xǁSEIRǁpredict_states__mutmut_54': xǁSEIRǁpredict_states__mutmut_54, 
        'xǁSEIRǁpredict_states__mutmut_55': xǁSEIRǁpredict_states__mutmut_55, 
        'xǁSEIRǁpredict_states__mutmut_56': xǁSEIRǁpredict_states__mutmut_56, 
        'xǁSEIRǁpredict_states__mutmut_57': xǁSEIRǁpredict_states__mutmut_57, 
        'xǁSEIRǁpredict_states__mutmut_58': xǁSEIRǁpredict_states__mutmut_58
    }
    xǁSEIRǁpredict_states__mutmut_orig.__name__ = 'xǁSEIRǁpredict_states'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSEIRǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁSEIRǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁSEIRǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters."""
        return {
            "XXtransmission_rateXX": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters."""
        return {
            "TRANSMISSION_RATE": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "XXtypeXX": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "TYPE": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "XXfloatXX",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "FLOAT",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "XXdefaultXX": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "DEFAULT": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "XXdescriptionXX": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "DESCRIPTION": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "XXThe rate of transmission of the contagion.XX",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "the rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "THE RATE OF TRANSMISSION OF THE CONTAGION.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_14(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "XXincubation_rateXX": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_15(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "INCUBATION_RATE": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_16(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "XXtypeXX": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_17(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "TYPE": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_18(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "XXfloatXX",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_19(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "FLOAT",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_20(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "XXdefaultXX": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_21(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "DEFAULT": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_22(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "XXdescriptionXX": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_23(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "DESCRIPTION": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_24(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "XXThe rate at which exposed individuals become infectious.XX",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_25(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "the rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_26(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "THE RATE AT WHICH EXPOSED INDIVIDUALS BECOME INFECTIOUS.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_27(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "XXrecovery_rateXX": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_28(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "RECOVERY_RATE": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_29(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "XXtypeXX": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_30(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "TYPE": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_31(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "XXfloatXX",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_32(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "FLOAT",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_33(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "XXdefaultXX": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_34(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "DEFAULT": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_35(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "XXdescriptionXX": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_36(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "DESCRIPTION": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_37(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "XXThe rate of recovery from the contagion.XX",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_38(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "the rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_39(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "THE RATE OF RECOVERY FROM THE CONTAGION.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_40(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "XXS0XX": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_41(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "s0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_42(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "XXtypeXX": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_43(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "TYPE": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_44(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "XXfloatXX",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_45(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "FLOAT",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_46(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "XXdefaultXX": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_47(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "DEFAULT": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_48(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 1000,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_49(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "XXdescriptionXX": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_50(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "DESCRIPTION": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_51(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "XXThe initial number of susceptible individuals.XX",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_52(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "the initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_53(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "THE INITIAL NUMBER OF SUSCEPTIBLE INDIVIDUALS.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_54(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "XXE0XX": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_55(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "e0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_56(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "XXtypeXX": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_57(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "TYPE": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_58(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "XXfloatXX",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_59(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "FLOAT",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_60(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "XXdefaultXX": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_61(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "DEFAULT": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_62(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_63(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "XXdescriptionXX": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_64(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "DESCRIPTION": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_65(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "XXThe initial number of exposed individuals.XX",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_66(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "the initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_67(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "THE INITIAL NUMBER OF EXPOSED INDIVIDUALS.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_68(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "XXI0XX": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_69(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "i0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_70(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "XXtypeXX": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_71(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "TYPE": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_72(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "XXfloatXX",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_73(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "FLOAT",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_74(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "XXdefaultXX": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_75(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "DEFAULT": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_76(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 2,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_77(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "XXdescriptionXX": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_78(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "DESCRIPTION": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_79(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "XXThe initial number of infectious individuals.XX",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_80(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "the initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_81(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "THE INITIAL NUMBER OF INFECTIOUS INDIVIDUALS.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_82(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "XXR0XX": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_83(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "r0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_84(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "XXtypeXX": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_85(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "TYPE": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_86(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "XXfloatXX",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_87(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "FLOAT",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_88(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "XXdefaultXX": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_89(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "DEFAULT": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_90(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_91(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "XXdescriptionXX": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_92(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "DESCRIPTION": "The initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_93(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "XXThe initial number of recovered individuals.XX",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_94(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "the initial number of recovered individuals.",
            },
        }

    def xǁSEIRǁget_parameters_schema__mutmut_95(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "THE INITIAL NUMBER OF RECOVERED INDIVIDUALS.",
            },
        }
    
    xǁSEIRǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSEIRǁget_parameters_schema__mutmut_1': xǁSEIRǁget_parameters_schema__mutmut_1, 
        'xǁSEIRǁget_parameters_schema__mutmut_2': xǁSEIRǁget_parameters_schema__mutmut_2, 
        'xǁSEIRǁget_parameters_schema__mutmut_3': xǁSEIRǁget_parameters_schema__mutmut_3, 
        'xǁSEIRǁget_parameters_schema__mutmut_4': xǁSEIRǁget_parameters_schema__mutmut_4, 
        'xǁSEIRǁget_parameters_schema__mutmut_5': xǁSEIRǁget_parameters_schema__mutmut_5, 
        'xǁSEIRǁget_parameters_schema__mutmut_6': xǁSEIRǁget_parameters_schema__mutmut_6, 
        'xǁSEIRǁget_parameters_schema__mutmut_7': xǁSEIRǁget_parameters_schema__mutmut_7, 
        'xǁSEIRǁget_parameters_schema__mutmut_8': xǁSEIRǁget_parameters_schema__mutmut_8, 
        'xǁSEIRǁget_parameters_schema__mutmut_9': xǁSEIRǁget_parameters_schema__mutmut_9, 
        'xǁSEIRǁget_parameters_schema__mutmut_10': xǁSEIRǁget_parameters_schema__mutmut_10, 
        'xǁSEIRǁget_parameters_schema__mutmut_11': xǁSEIRǁget_parameters_schema__mutmut_11, 
        'xǁSEIRǁget_parameters_schema__mutmut_12': xǁSEIRǁget_parameters_schema__mutmut_12, 
        'xǁSEIRǁget_parameters_schema__mutmut_13': xǁSEIRǁget_parameters_schema__mutmut_13, 
        'xǁSEIRǁget_parameters_schema__mutmut_14': xǁSEIRǁget_parameters_schema__mutmut_14, 
        'xǁSEIRǁget_parameters_schema__mutmut_15': xǁSEIRǁget_parameters_schema__mutmut_15, 
        'xǁSEIRǁget_parameters_schema__mutmut_16': xǁSEIRǁget_parameters_schema__mutmut_16, 
        'xǁSEIRǁget_parameters_schema__mutmut_17': xǁSEIRǁget_parameters_schema__mutmut_17, 
        'xǁSEIRǁget_parameters_schema__mutmut_18': xǁSEIRǁget_parameters_schema__mutmut_18, 
        'xǁSEIRǁget_parameters_schema__mutmut_19': xǁSEIRǁget_parameters_schema__mutmut_19, 
        'xǁSEIRǁget_parameters_schema__mutmut_20': xǁSEIRǁget_parameters_schema__mutmut_20, 
        'xǁSEIRǁget_parameters_schema__mutmut_21': xǁSEIRǁget_parameters_schema__mutmut_21, 
        'xǁSEIRǁget_parameters_schema__mutmut_22': xǁSEIRǁget_parameters_schema__mutmut_22, 
        'xǁSEIRǁget_parameters_schema__mutmut_23': xǁSEIRǁget_parameters_schema__mutmut_23, 
        'xǁSEIRǁget_parameters_schema__mutmut_24': xǁSEIRǁget_parameters_schema__mutmut_24, 
        'xǁSEIRǁget_parameters_schema__mutmut_25': xǁSEIRǁget_parameters_schema__mutmut_25, 
        'xǁSEIRǁget_parameters_schema__mutmut_26': xǁSEIRǁget_parameters_schema__mutmut_26, 
        'xǁSEIRǁget_parameters_schema__mutmut_27': xǁSEIRǁget_parameters_schema__mutmut_27, 
        'xǁSEIRǁget_parameters_schema__mutmut_28': xǁSEIRǁget_parameters_schema__mutmut_28, 
        'xǁSEIRǁget_parameters_schema__mutmut_29': xǁSEIRǁget_parameters_schema__mutmut_29, 
        'xǁSEIRǁget_parameters_schema__mutmut_30': xǁSEIRǁget_parameters_schema__mutmut_30, 
        'xǁSEIRǁget_parameters_schema__mutmut_31': xǁSEIRǁget_parameters_schema__mutmut_31, 
        'xǁSEIRǁget_parameters_schema__mutmut_32': xǁSEIRǁget_parameters_schema__mutmut_32, 
        'xǁSEIRǁget_parameters_schema__mutmut_33': xǁSEIRǁget_parameters_schema__mutmut_33, 
        'xǁSEIRǁget_parameters_schema__mutmut_34': xǁSEIRǁget_parameters_schema__mutmut_34, 
        'xǁSEIRǁget_parameters_schema__mutmut_35': xǁSEIRǁget_parameters_schema__mutmut_35, 
        'xǁSEIRǁget_parameters_schema__mutmut_36': xǁSEIRǁget_parameters_schema__mutmut_36, 
        'xǁSEIRǁget_parameters_schema__mutmut_37': xǁSEIRǁget_parameters_schema__mutmut_37, 
        'xǁSEIRǁget_parameters_schema__mutmut_38': xǁSEIRǁget_parameters_schema__mutmut_38, 
        'xǁSEIRǁget_parameters_schema__mutmut_39': xǁSEIRǁget_parameters_schema__mutmut_39, 
        'xǁSEIRǁget_parameters_schema__mutmut_40': xǁSEIRǁget_parameters_schema__mutmut_40, 
        'xǁSEIRǁget_parameters_schema__mutmut_41': xǁSEIRǁget_parameters_schema__mutmut_41, 
        'xǁSEIRǁget_parameters_schema__mutmut_42': xǁSEIRǁget_parameters_schema__mutmut_42, 
        'xǁSEIRǁget_parameters_schema__mutmut_43': xǁSEIRǁget_parameters_schema__mutmut_43, 
        'xǁSEIRǁget_parameters_schema__mutmut_44': xǁSEIRǁget_parameters_schema__mutmut_44, 
        'xǁSEIRǁget_parameters_schema__mutmut_45': xǁSEIRǁget_parameters_schema__mutmut_45, 
        'xǁSEIRǁget_parameters_schema__mutmut_46': xǁSEIRǁget_parameters_schema__mutmut_46, 
        'xǁSEIRǁget_parameters_schema__mutmut_47': xǁSEIRǁget_parameters_schema__mutmut_47, 
        'xǁSEIRǁget_parameters_schema__mutmut_48': xǁSEIRǁget_parameters_schema__mutmut_48, 
        'xǁSEIRǁget_parameters_schema__mutmut_49': xǁSEIRǁget_parameters_schema__mutmut_49, 
        'xǁSEIRǁget_parameters_schema__mutmut_50': xǁSEIRǁget_parameters_schema__mutmut_50, 
        'xǁSEIRǁget_parameters_schema__mutmut_51': xǁSEIRǁget_parameters_schema__mutmut_51, 
        'xǁSEIRǁget_parameters_schema__mutmut_52': xǁSEIRǁget_parameters_schema__mutmut_52, 
        'xǁSEIRǁget_parameters_schema__mutmut_53': xǁSEIRǁget_parameters_schema__mutmut_53, 
        'xǁSEIRǁget_parameters_schema__mutmut_54': xǁSEIRǁget_parameters_schema__mutmut_54, 
        'xǁSEIRǁget_parameters_schema__mutmut_55': xǁSEIRǁget_parameters_schema__mutmut_55, 
        'xǁSEIRǁget_parameters_schema__mutmut_56': xǁSEIRǁget_parameters_schema__mutmut_56, 
        'xǁSEIRǁget_parameters_schema__mutmut_57': xǁSEIRǁget_parameters_schema__mutmut_57, 
        'xǁSEIRǁget_parameters_schema__mutmut_58': xǁSEIRǁget_parameters_schema__mutmut_58, 
        'xǁSEIRǁget_parameters_schema__mutmut_59': xǁSEIRǁget_parameters_schema__mutmut_59, 
        'xǁSEIRǁget_parameters_schema__mutmut_60': xǁSEIRǁget_parameters_schema__mutmut_60, 
        'xǁSEIRǁget_parameters_schema__mutmut_61': xǁSEIRǁget_parameters_schema__mutmut_61, 
        'xǁSEIRǁget_parameters_schema__mutmut_62': xǁSEIRǁget_parameters_schema__mutmut_62, 
        'xǁSEIRǁget_parameters_schema__mutmut_63': xǁSEIRǁget_parameters_schema__mutmut_63, 
        'xǁSEIRǁget_parameters_schema__mutmut_64': xǁSEIRǁget_parameters_schema__mutmut_64, 
        'xǁSEIRǁget_parameters_schema__mutmut_65': xǁSEIRǁget_parameters_schema__mutmut_65, 
        'xǁSEIRǁget_parameters_schema__mutmut_66': xǁSEIRǁget_parameters_schema__mutmut_66, 
        'xǁSEIRǁget_parameters_schema__mutmut_67': xǁSEIRǁget_parameters_schema__mutmut_67, 
        'xǁSEIRǁget_parameters_schema__mutmut_68': xǁSEIRǁget_parameters_schema__mutmut_68, 
        'xǁSEIRǁget_parameters_schema__mutmut_69': xǁSEIRǁget_parameters_schema__mutmut_69, 
        'xǁSEIRǁget_parameters_schema__mutmut_70': xǁSEIRǁget_parameters_schema__mutmut_70, 
        'xǁSEIRǁget_parameters_schema__mutmut_71': xǁSEIRǁget_parameters_schema__mutmut_71, 
        'xǁSEIRǁget_parameters_schema__mutmut_72': xǁSEIRǁget_parameters_schema__mutmut_72, 
        'xǁSEIRǁget_parameters_schema__mutmut_73': xǁSEIRǁget_parameters_schema__mutmut_73, 
        'xǁSEIRǁget_parameters_schema__mutmut_74': xǁSEIRǁget_parameters_schema__mutmut_74, 
        'xǁSEIRǁget_parameters_schema__mutmut_75': xǁSEIRǁget_parameters_schema__mutmut_75, 
        'xǁSEIRǁget_parameters_schema__mutmut_76': xǁSEIRǁget_parameters_schema__mutmut_76, 
        'xǁSEIRǁget_parameters_schema__mutmut_77': xǁSEIRǁget_parameters_schema__mutmut_77, 
        'xǁSEIRǁget_parameters_schema__mutmut_78': xǁSEIRǁget_parameters_schema__mutmut_78, 
        'xǁSEIRǁget_parameters_schema__mutmut_79': xǁSEIRǁget_parameters_schema__mutmut_79, 
        'xǁSEIRǁget_parameters_schema__mutmut_80': xǁSEIRǁget_parameters_schema__mutmut_80, 
        'xǁSEIRǁget_parameters_schema__mutmut_81': xǁSEIRǁget_parameters_schema__mutmut_81, 
        'xǁSEIRǁget_parameters_schema__mutmut_82': xǁSEIRǁget_parameters_schema__mutmut_82, 
        'xǁSEIRǁget_parameters_schema__mutmut_83': xǁSEIRǁget_parameters_schema__mutmut_83, 
        'xǁSEIRǁget_parameters_schema__mutmut_84': xǁSEIRǁget_parameters_schema__mutmut_84, 
        'xǁSEIRǁget_parameters_schema__mutmut_85': xǁSEIRǁget_parameters_schema__mutmut_85, 
        'xǁSEIRǁget_parameters_schema__mutmut_86': xǁSEIRǁget_parameters_schema__mutmut_86, 
        'xǁSEIRǁget_parameters_schema__mutmut_87': xǁSEIRǁget_parameters_schema__mutmut_87, 
        'xǁSEIRǁget_parameters_schema__mutmut_88': xǁSEIRǁget_parameters_schema__mutmut_88, 
        'xǁSEIRǁget_parameters_schema__mutmut_89': xǁSEIRǁget_parameters_schema__mutmut_89, 
        'xǁSEIRǁget_parameters_schema__mutmut_90': xǁSEIRǁget_parameters_schema__mutmut_90, 
        'xǁSEIRǁget_parameters_schema__mutmut_91': xǁSEIRǁget_parameters_schema__mutmut_91, 
        'xǁSEIRǁget_parameters_schema__mutmut_92': xǁSEIRǁget_parameters_schema__mutmut_92, 
        'xǁSEIRǁget_parameters_schema__mutmut_93': xǁSEIRǁget_parameters_schema__mutmut_93, 
        'xǁSEIRǁget_parameters_schema__mutmut_94': xǁSEIRǁget_parameters_schema__mutmut_94, 
        'xǁSEIRǁget_parameters_schema__mutmut_95': xǁSEIRǁget_parameters_schema__mutmut_95
    }
    xǁSEIRǁget_parameters_schema__mutmut_orig.__name__ = 'xǁSEIRǁget_parameters_schema'
