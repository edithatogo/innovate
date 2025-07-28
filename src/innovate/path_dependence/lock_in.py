import numpy as np
from typing import Sequence, Dict
from innovate.base.base import DiffusionModel
from scipy.integrate import odeint

class LockInModel(DiffusionModel):
    """
    A simple model demonstrating path dependence and lock-in effects
    between two competing technologies.

    The model simulates two technologies where the growth rate of each
    is positively influenced by its own installed base (network effects)
    and negatively by the competitor's.
    """

    def __init__(self):
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        return [
            "alpha1",  # Intrinsic growth rate of Tech 1
            "alpha2",  # Intrinsic growth rate of Tech 2
            "beta1",   # Network effect strength for Tech 1
            "beta2",   # Network effect strength for Tech 2
            "gamma1",  # Negative influence of Tech 2 on Tech 1
            "gamma2",  # Negative influence of Tech 1 on Tech 2
            "m",       # Total market potential (assumed shared)
        ]

    def initial_guesses(self, t: Sequence[float], y: np.ndarray) -> Dict[str, float]:
        # y is expected to be a 2D array: [adoptions_tech1, adoptions_tech2]
        max_y = np.max(y)
        return {
            "alpha1": 0.1,
            "alpha2": 0.1,
            "beta1": 0.01,
            "beta2": 0.01,
            "gamma1": 0.001,
            "gamma2": 0.001,
            "m": max_y * 1.5 if max_y > 0 else 1000.0,
        }

    def bounds(self, t: Sequence[float], y: np.ndarray) -> Dict[str, tuple]:
        max_y = np.max(y)
        return {
            "alpha1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta1": (0, np.inf),
            "beta2": (0, np.inf),
            "gamma1": (0, np.inf),
            "gamma2": (0, np.inf),
            "m": (max_y, np.inf),
        }

    def differential_equation(self, y_current: Sequence[float], t_current: float, alpha1, alpha2, beta1, beta2, gamma1, gamma2, m) -> Sequence[float]:
        n1, n2 = y_current

        # Ensure populations are non-negative and do not exceed market potential
        n1 = max(0, min(n1, m))
        n2 = max(0, min(n2, m))

        # Simple logistic-like growth with network effects and competition
        dn1_dt = alpha1 * n1 * (1 - (n1 + n2) / m) + beta1 * n1 * (n1 / m) - gamma1 * n1 * (n2 / m)
        dn2_dt = alpha2 * n2 * (1 - (n1 + n2) / m) + beta2 * n2 * (n2 / m) - gamma2 * n2 * (n1 / m)
        
        return [dn1_dt, dn2_dt]

    def predict(self, t: Sequence[float], y0: Sequence[float]) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        sol = odeint(
            self.differential_equation,
            y0,
            t,
            args=tuple(self._params.values()),
        )
        sol = np.maximum(0, sol)
        m = self._params.get("m", np.inf)
        sol = np.minimum(sol, m)
        return sol

    def fit(self, t: Sequence[float], y: np.ndarray, **kwargs):
        from scipy.optimize import minimize

        y = np.array(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("`y` must be a 2D array with two columns (for two technologies).")

        y0 = y[0, :]

        def objective(params, t, y_obs):
            self.params_ = dict(zip(self.param_names, params))
            y_pred = self.predict(t, y0)
            return np.sum((y_obs - y_pred) ** 2)

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
        
        # To get adoption rates, we need to calculate the derivative at each point
        # This is more complex for ODEs. For simplicity, we can approximate it
        # by taking the difference between cumulative predictions.
        cumulative_predictions = self.predict(t, y0)
        # Calculate the difference between consecutive cumulative values
        # Pad with initial y0 for the first step's rate
        rates = np.diff(cumulative_predictions, axis=0)
        # The first rate can be approximated as the first cumulative value if starting from 0
        # Or, more accurately, by evaluating the RHS at t[0] with y0
        initial_rates = self.differential_equation(y0, t[0], *self._params.values())
        return np.vstack([initial_rates, rates])