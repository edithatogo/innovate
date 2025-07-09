# src/innovate/ecosystem/complementary_goods.py

import numpy as np
from typing import Sequence, Dict
from innovate.base.base import DiffusionModel

class ComplementaryGoodsModel(DiffusionModel):
    """
    A model for the diffusion of two complementary goods, where the
    adoption of each good is positively influenced by the adoption of the
    other.
    """

    def __init__(self):
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        return [
            "k1",  # Intrinsic growth rate of good 1
            "k2",  # Intrinsic growth rate of good 2
            "c1",  # Influence of good 2 on good 1
            "c2",  # Influence of good 1 on good 2
        ]

    def predict(self, t: Sequence[float], y0: Sequence[float]) -> np.ndarray:
        """
        Predicts the adoption of both goods over time.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        from scipy.integrate import odeint

        k1, k2, c1, c2 = (
            self._params["k1"],
            self._params["k2"],
            self._params["c1"],
            self._params["c2"],
        )

        def system(y, t):
            y1, y2 = y
            dy1_dt = k1 * y1 * (1 - y1) + c1 * y1 * y2
            dy2_dt = k2 * y2 * (1 - y2) + c2 * y1 * y2
            return [dy1_dt, dy2_dt]

        solution = odeint(system, y0, t)
        return solution

    def fit(self, t: Sequence[float], y: np.ndarray, **kwargs):
        """
        Fits the model to the data.
        """
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns.")

        y0 = y[0, :]

        def objective(params, t, y):
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return np.sum((y - y_pred) ** 2)

        initial_params = list(self.initial_guesses(t, y).values())
        param_bounds = list(self.bounds(t, y).values())

        result = minimize(
            objective,
            initial_params,
            args=(t, y),
            bounds=param_bounds,
            method='L-BFGS-B',
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        self.params_ = dict(zip(self.param_names, result.x))
        return self

    def initial_guesses(self, t: Sequence[float], y: np.ndarray) -> Dict[str, float]:
        return {"k1": 0.1, "k2": 0.1, "c1": 0.01, "c2": 0.01}

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        return {
            "k1": (0, np.inf),
            "k2": (0, np.inf),
            "c1": (0, np.inf),
            "c2": (0, np.inf),
        }

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value

    def score(self, t: Sequence[float], y: np.ndarray) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")
        y_pred = self.predict(t, y[0, :])
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict_adoption_rate(self, t: Sequence[float], y0: Sequence[float]) -> np.ndarray:
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