from .base import DiffusionModel, Self
from ..backend import current_backend as B
from typing import Sequence, Dict
import numpy as np

class GompertzModel(DiffusionModel):
    """Implementation of the Gompertz Diffusion Model."""

    def __init__(self):
        self._a: float = None
        self._b: float = None
        self._c: float = None

    @B.jit
    def _gompertz_cumulative(self, t, a, b, c):
        """The closed-form solution for cumulative adoptions in the Gompertz model."""
        return a * B.exp(-b * B.exp(-c * t))

    def fit(self, t: Sequence[float], y: Sequence[float]) -> Self:
        from scipy.optimize import curve_fit

        t_arr = np.array(t)
        y_arr = np.array(y)

        # Initial guesses for a, b, c
        # a: upper asymptote (ultimate market potential)
        # b: displacement on the x-axis (related to inflection point)
        # c: growth rate
        initial_a = np.max(y_arr) * 1.1
        initial_b = 1.0
        initial_c = 0.1

        # Bounds for parameters (a, b, c)
        # a must be > max(y), b and c must be > 0
        bounds = ([np.max(y_arr), 1e-6, 1e-6], [np.inf, np.inf, np.inf])

        try:
            params, _ = curve_fit(self._gompertz_cumulative, t_arr, y_arr, 
                                  p0=[initial_a, initial_b, initial_c], 
                                  bounds=bounds,
                                  maxfev=5000)
            self._a, self._b, self._c = params
        except RuntimeError as e:
            raise RuntimeError(f"GompertzModel fitting failed: {e}. Try different initial guesses or check data.")

        return self

    def predict(self, t: Sequence[float]) -> Sequence[float]:
        if self._a is None or self._b is None or self._c is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = B.array(t)
        return self._gompertz_cumulative(t_arr, self._a, self._b, self._c)

    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        if self._a is None or self._b is None or self._c is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def params_(self) -> Dict[str, float]:
        if self._a is None or self._b is None or self._c is None:
            return {}
        return {"a": self._a, "b": self._b, "c": self._c}

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        if self._a is None or self._b is None or self._c is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = B.array(t)
        exp_term_c = B.exp(-self._c * t_arr)
        return self._a * self._b * self._c * B.exp(-self._b * exp_term_c - self._c * t_arr)
