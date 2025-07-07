from .base import DiffusionModel, Self
from ..backend import current_backend as B
from typing import Sequence, Dict
import numpy as np

class LogisticModel(DiffusionModel):
    """Implementation of the Logistic Diffusion Model."""

    def __init__(self):
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        return ["L", "k", "x0"]

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        return {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        return {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }

    @staticmethod
    def cumulative_adoption(t, L, k, x0):
        from ..backend import current_backend as B
        """The closed-form solution for cumulative adoptions in the Logistic model."""
        return L / (1 + B.exp(-k * (t - x0)))

    def predict(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = B.array(t)
        return self.cumulative_adoption(t_arr, **self._params)

    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = B.array(t)
        L, k, x0 = self._params["L"], self._params["k"], self._params["x0"]
        exp_term = B.exp(-k * (t_arr - x0))
        return (L * k * exp_term) / ((1 + exp_term) ** 2)
