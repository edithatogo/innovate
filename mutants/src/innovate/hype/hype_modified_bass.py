# src/innovate/hype/hype_modified_bass.py

from collections.abc import Sequence

import numpy as np

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


class HypeModifiedBassModel:
    """A modified Bass model where the adoption parameters (p and q) are
    influenced by a time-varying hype function.
    """

    def __init__(self, bass_model: BassModel, hype_model: HypeCycleModel):
        args = [bass_model, hype_model]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHypeModifiedBassModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁHypeModifiedBassModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁHypeModifiedBassModelǁ__init____mutmut_orig(self, bass_model: BassModel, hype_model: HypeCycleModel):
        self.bass_model = bass_model
        self.hype_model = hype_model

    def xǁHypeModifiedBassModelǁ__init____mutmut_1(self, bass_model: BassModel, hype_model: HypeCycleModel):
        self.bass_model = None
        self.hype_model = hype_model

    def xǁHypeModifiedBassModelǁ__init____mutmut_2(self, bass_model: BassModel, hype_model: HypeCycleModel):
        self.bass_model = bass_model
        self.hype_model = None
    
    xǁHypeModifiedBassModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHypeModifiedBassModelǁ__init____mutmut_1': xǁHypeModifiedBassModelǁ__init____mutmut_1, 
        'xǁHypeModifiedBassModelǁ__init____mutmut_2': xǁHypeModifiedBassModelǁ__init____mutmut_2
    }
    xǁHypeModifiedBassModelǁ__init____mutmut_orig.__name__ = 'xǁHypeModifiedBassModelǁ__init__'

    def predict(self, t: Sequence[float], y0: float) -> np.ndarray:
        args = [t, y0]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHypeModifiedBassModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁHypeModifiedBassModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁHypeModifiedBassModelǁpredict__mutmut_orig(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_1(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ and not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_2(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_3(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_4(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                None,
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_5(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "XXBoth the Bass and Hype models must have parameters set.XX",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_6(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "both the bass and hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_7(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "BOTH THE BASS AND HYPE MODELS MUST HAVE PARAMETERS SET.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_8(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = None
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_9(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(None)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_10(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = None
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_11(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["XXpXX"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_12(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["P"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_13(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = None
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_14(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["XXqXX"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_15(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["Q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_16(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = None

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_17(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["XXmXX"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_18(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["M"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_19(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) / (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_20(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t - q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_21(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y * m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_22(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t / y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_23(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m + y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_24(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = None
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_25(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(None, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_26(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=None)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_27(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_28(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, )
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_29(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = None

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_30(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[1] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_31(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(None, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_32(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, None):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_33(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_34(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, ):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_35(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(2, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_36(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = None
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_37(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base / (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_38(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 - hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_39(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (2 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_40(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i + 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_41(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 2])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_42(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = None

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_43(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base / (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_44(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 - hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_45(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (2 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_46(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i + 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_47(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 2])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_48(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = None
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_49(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                None,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_50(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                None,
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_51(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                None,
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_52(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=None,
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_53(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_54(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_55(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_56(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_57(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i + 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_58(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 2],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_59(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i + 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_60(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 2], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[1]

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_61(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = None

        return y

    def xǁHypeModifiedBassModelǁpredict__mutmut_62(self, t: Sequence[float], y0: float) -> np.ndarray:
        """Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set.",
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential,
                y[i - 1],
                [t[i - 1], t[i]],
                args=(p_t, q_t),
            )
            y[i] = y_step[2]

        return y
    
    xǁHypeModifiedBassModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHypeModifiedBassModelǁpredict__mutmut_1': xǁHypeModifiedBassModelǁpredict__mutmut_1, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_2': xǁHypeModifiedBassModelǁpredict__mutmut_2, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_3': xǁHypeModifiedBassModelǁpredict__mutmut_3, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_4': xǁHypeModifiedBassModelǁpredict__mutmut_4, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_5': xǁHypeModifiedBassModelǁpredict__mutmut_5, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_6': xǁHypeModifiedBassModelǁpredict__mutmut_6, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_7': xǁHypeModifiedBassModelǁpredict__mutmut_7, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_8': xǁHypeModifiedBassModelǁpredict__mutmut_8, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_9': xǁHypeModifiedBassModelǁpredict__mutmut_9, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_10': xǁHypeModifiedBassModelǁpredict__mutmut_10, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_11': xǁHypeModifiedBassModelǁpredict__mutmut_11, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_12': xǁHypeModifiedBassModelǁpredict__mutmut_12, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_13': xǁHypeModifiedBassModelǁpredict__mutmut_13, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_14': xǁHypeModifiedBassModelǁpredict__mutmut_14, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_15': xǁHypeModifiedBassModelǁpredict__mutmut_15, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_16': xǁHypeModifiedBassModelǁpredict__mutmut_16, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_17': xǁHypeModifiedBassModelǁpredict__mutmut_17, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_18': xǁHypeModifiedBassModelǁpredict__mutmut_18, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_19': xǁHypeModifiedBassModelǁpredict__mutmut_19, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_20': xǁHypeModifiedBassModelǁpredict__mutmut_20, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_21': xǁHypeModifiedBassModelǁpredict__mutmut_21, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_22': xǁHypeModifiedBassModelǁpredict__mutmut_22, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_23': xǁHypeModifiedBassModelǁpredict__mutmut_23, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_24': xǁHypeModifiedBassModelǁpredict__mutmut_24, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_25': xǁHypeModifiedBassModelǁpredict__mutmut_25, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_26': xǁHypeModifiedBassModelǁpredict__mutmut_26, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_27': xǁHypeModifiedBassModelǁpredict__mutmut_27, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_28': xǁHypeModifiedBassModelǁpredict__mutmut_28, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_29': xǁHypeModifiedBassModelǁpredict__mutmut_29, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_30': xǁHypeModifiedBassModelǁpredict__mutmut_30, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_31': xǁHypeModifiedBassModelǁpredict__mutmut_31, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_32': xǁHypeModifiedBassModelǁpredict__mutmut_32, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_33': xǁHypeModifiedBassModelǁpredict__mutmut_33, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_34': xǁHypeModifiedBassModelǁpredict__mutmut_34, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_35': xǁHypeModifiedBassModelǁpredict__mutmut_35, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_36': xǁHypeModifiedBassModelǁpredict__mutmut_36, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_37': xǁHypeModifiedBassModelǁpredict__mutmut_37, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_38': xǁHypeModifiedBassModelǁpredict__mutmut_38, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_39': xǁHypeModifiedBassModelǁpredict__mutmut_39, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_40': xǁHypeModifiedBassModelǁpredict__mutmut_40, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_41': xǁHypeModifiedBassModelǁpredict__mutmut_41, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_42': xǁHypeModifiedBassModelǁpredict__mutmut_42, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_43': xǁHypeModifiedBassModelǁpredict__mutmut_43, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_44': xǁHypeModifiedBassModelǁpredict__mutmut_44, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_45': xǁHypeModifiedBassModelǁpredict__mutmut_45, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_46': xǁHypeModifiedBassModelǁpredict__mutmut_46, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_47': xǁHypeModifiedBassModelǁpredict__mutmut_47, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_48': xǁHypeModifiedBassModelǁpredict__mutmut_48, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_49': xǁHypeModifiedBassModelǁpredict__mutmut_49, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_50': xǁHypeModifiedBassModelǁpredict__mutmut_50, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_51': xǁHypeModifiedBassModelǁpredict__mutmut_51, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_52': xǁHypeModifiedBassModelǁpredict__mutmut_52, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_53': xǁHypeModifiedBassModelǁpredict__mutmut_53, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_54': xǁHypeModifiedBassModelǁpredict__mutmut_54, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_55': xǁHypeModifiedBassModelǁpredict__mutmut_55, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_56': xǁHypeModifiedBassModelǁpredict__mutmut_56, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_57': xǁHypeModifiedBassModelǁpredict__mutmut_57, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_58': xǁHypeModifiedBassModelǁpredict__mutmut_58, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_59': xǁHypeModifiedBassModelǁpredict__mutmut_59, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_60': xǁHypeModifiedBassModelǁpredict__mutmut_60, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_61': xǁHypeModifiedBassModelǁpredict__mutmut_61, 
        'xǁHypeModifiedBassModelǁpredict__mutmut_62': xǁHypeModifiedBassModelǁpredict__mutmut_62
    }
    xǁHypeModifiedBassModelǁpredict__mutmut_orig.__name__ = 'xǁHypeModifiedBassModelǁpredict'
