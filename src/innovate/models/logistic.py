from .base import DiffusionModel, Self
from ..backend import current_backend as B
from typing import Sequence, Dict
import numpy as np

class LogisticModel(DiffusionModel):
    """Implementation of the Logistic Diffusion Model."""

    def __init__(self):
        self._L: float = None
        self._k: float = None
        self._x0: float = None

    @B.jit
    def _logistic_cumulative(self, t, L, k, x0):
        """The closed-form solution for cumulative adoptions in the Logistic model."""
        return L / (1 + B.exp(-k * (t - x0)))

    def fit(self, t: Sequence[float], y: Sequence[float]) -> Self:
        from scipy.optimize import curve_fit

        t_arr = B.array(t)
        y_arr = B.array(y)

        # Initial guesses for L, k, x0
        # L: the curve's maximum value (ultimate market potential)
        # k: the logistic growth rate or steepness of the curve
        # x0: the x-value of the sigmoid's midpoint (inflection point)
        initial_L = B.max(y_arr) * 1.1
        initial_k = 0.1
        initial_x0 = B.median(t_arr)

        # Bounds for parameters (L, k, x0)
        # L must be > max(y), k > 0
        bounds = ([B.max(y_arr), 1e-6, -np.inf], [np.inf, np.inf, np.inf])

        try:
            params, _ = curve_fit(self._logistic_cumulative, t_arr, y_arr, 
                                  p0=[initial_L, initial_k, initial_x0], 
                                  bounds=bounds,
                                  maxfev=5000)
            self._L, self._k, self._x0 = params
        except RuntimeError as e:
            raise RuntimeError(f"LogisticModel fitting failed: {e}. Try different initial guesses or check data.")

        return self

    def predict(self, t: Sequence[float]) -> Sequence[float]:
        if self._L is None or self._k is None or self._x0 is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = B.array(t)
        return self._logistic_cumulative(t_arr, self._L, self._k, self._x0)

    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        if self._L is None or self._k is None or self._x0 is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def params_(self) -> Dict[str, float]:
        if self._L is None or self._k is None or self._x0 is None:
            return {}
        return {"L": self._L, "k": self._k, "x0": self._x0}

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        if self._L is None or self._k is None or self._x0 is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = B.array(t)
        exp_term = B.exp(-self._k * (t_arr - self._x0))
        return (self._L * self._k * exp_term) / ((1 + exp_term) ** 2)

    def _get_fitted_params_as_array(self):
        if self._L is None or self._k is None or self._x0 is None:
            raise RuntimeError("Model has not been fitted yet. Parameters are not set.")
        return B.array([self._L, self._k, self._x0])

    def _get_initial_params_as_array(self, t: Sequence[float], y: Sequence[float]):
        # Initial guesses for L, k, x0
        initial_L = B.max(B.array(y)) * 1.1
        initial_k = 0.1
        initial_x0 = B.median(B.array(t))
        return B.array([initial_L, initial_k, initial_x0])

    def _get_fitted_params_as_array(self):
        if self._L is None or self._k is None or self._x0 is None:
            raise RuntimeError("Model has not been fitted yet. Parameters are not set.")
        return B.array([self._L, self._k, self._x0])

    def _get_initial_params_as_array(self, t: Sequence[float], y: Sequence[float]):
        # Initial guesses for L, k, x0
        initial_L = B.max(B.array(y)) * 1.1
        initial_k = 0.1
        initial_x0 = B.median(B.array(t))
        return B.array([initial_L, initial_k, initial_x0])
