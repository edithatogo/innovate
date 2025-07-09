from innovate.base.base import DiffusionModel, Self
from innovate.backend import current_backend as B
from typing import Sequence, Dict
import numpy as np

class GompertzModel(DiffusionModel):
    """Implementation of the Gompertz Diffusion Model."""

    def __init__(self):
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        return ["a", "b", "c"]

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        return {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        return {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }

    @staticmethod
    def cumulative_adoption(t, a, b, c):
        from innovate.backend import current_backend as B
        """The closed-form solution for cumulative adoptions in the Gompertz model."""
        return a * B.exp(-b * B.exp(-c * t))

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
        a, b, c = self._params["a"], self._params["b"], self._params["c"]
        exp_term_c = B.exp(-c * t_arr)
        return a * b * c * B.exp(-b * exp_term_c - c * t_arr)
