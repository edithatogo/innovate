from innovate.base.base import DiffusionModel, Self
from innovate.backend import current_backend as B
from typing import Sequence, Dict
import numpy as np

class BassModel(DiffusionModel):
    """Implementation of the Bass Diffusion Model."""

    def __init__(self):
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        return ["p", "q", "m"]

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        return {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y) * 1.1,
        }

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        return {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y), np.inf),
        }

    @staticmethod
    def cumulative_adoption(t, p, q, m):
        from innovate.backend import current_backend as B
        """The closed-form solution for cumulative adoptions in the Bass model."""
        exp_term = B.exp(-(p + q) * t)
        return m * (1 - exp_term) / (1 + (q / p) * exp_term)

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
        N = self.predict(t_arr)
        p, q, m = self._params["p"], self._params["q"], self._params["m"]
        return (p + q * (N / m)) * (m - N)
