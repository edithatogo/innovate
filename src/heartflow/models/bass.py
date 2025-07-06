from .base import DiffusionModel, Self
from ..backend import current_backend as B
from typing import Sequence, Dict
import numpy as np

class BassModel(DiffusionModel):
    """Implementation of the Bass Diffusion Model."""

    def __init__(self):
        self._p: float = None
        self._q: float = None
        self._m: float = None

    def _bass_ode(self, N, t, p, q, m):
        """The differential equation for the Bass model."""
        return (p + q * (N / m)) * (m - N)

    @B.jit
    def _bass_cumulative(self, t, p, q, m):
        """The closed-form solution for cumulative adoptions in the Bass model."""
        exp_term = B.exp(-(p + q) * t)
        return m * (1 - exp_term) / (1 + (q / p) * exp_term)

    def fit(self, t: Sequence[float], y: Sequence[float]) -> Self:
        from scipy.optimize import curve_fit

        t_arr = np.array(t)
        y_arr = np.array(y)

        # Initial guesses for p, q, m
        # p: coefficient of innovation (external influence)
        # q: coefficient of imitation (internal influence)
        # m: ultimate market potential
        # These initial guesses are crucial for successful fitting.
        # A common heuristic for m is the maximum observed value or slightly higher.
        # For p and q, small positive values are typical.
        initial_m = np.max(y_arr) * 1.1 # Slightly above max observed
        initial_p = 0.001
        initial_q = 0.1

        # Bounds for parameters (p, q, m)
        # p, q must be > 0. m must be > max(y)
        bounds = ([1e-6, 1e-6, np.max(y_arr)], [0.1, 1.0, np.inf])

        try:
            params, _ = curve_fit(self._bass_cumulative, t_arr, y_arr, 
                                  p0=[initial_p, initial_q, initial_m], 
                                  bounds=bounds,
                                  maxfev=5000)
            self._p, self._q, self._m = params
        except RuntimeError as e:
            raise RuntimeError(f"BassModel fitting failed: {e}. Try different initial guesses or check data.")

        return self

    def predict(self, t: Sequence[float]) -> Sequence[float]:
        if self._p is None or self._q is None or self._m is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = B.array(t)
        return self._bass_cumulative(t_arr, self._p, self._q, self._m)

    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        if self._p is None or self._q is None or self._m is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def params_(self) -> Dict[str, float]:
        if self._p is None or self._q is None or self._m is None:
            return {}
        return {"p": self._p, "q": self._q, "m": self._m}
