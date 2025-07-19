# src/innovate/substitute/fisher_pry.py

from innovate.base.base import DiffusionModel
from innovate.backend import current_backend as B
from typing import Sequence, Dict
import numpy as np

class FisherPryModel(DiffusionModel):
    """
    Implementation of the Fisher-Pry model for technology substitution.

    This model assumes that the substitution of a new technology for an old one
    follows a logistic growth curve. The model tracks the market share
    fraction of the new technology.
    """

    def __init__(self):
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        """Returns the names of the model parameters: alpha and t0."""
        return ["alpha", "t0"]

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        """
        Provides initial guesses for the model parameters.
        - t0 is estimated as the time at which the market share is closest to 50%.
        - alpha is estimated from a linearization of the logistic function.
        """
        y_arr = B.array(y)
        t_arr = B.array(t)

        # Estimate t0 as the time when market share is closest to 0.5
        t0_guess = t_arr[B.argmin(B.abs(y_arr - 0.5))]

        # Linearize the logistic equation: log(y / (1 - y)) = alpha * (t - t0)
        # To avoid division by zero or log of zero, we clip y
        y_clipped = B.clip(y_arr, 1e-6, 1 - 1e-6)
        linearized_y = B.log(y_clipped / (1 - y_clipped))

        # Perform a linear regression to find the slope (alpha)
        try:
            # Using polyfit for a simple linear regression
            slope, _ = np.polyfit(t_arr, linearized_y, 1)
            alpha_guess = max(0, slope) # Ensure alpha is non-negative
        except (np.linalg.LinAlgError, ValueError):
            alpha_guess = 0.5 # Fallback value

        return {
            "alpha": alpha_guess,
            "t0": t0_guess,
        }

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        """Returns bounds for the model parameters."""
        t_min, t_max = B.min(t), B.max(t)
        t_range = t_max - t_min
        return {
            "alpha": (0, np.inf),
            "t0": (t_min - t_range, t_max + t_range),
        }

    def differential_equation(self, y, t, alpha, t0):
        """The differential equation for the Fisher-Pry model."""
        return alpha * y * (1 - y)

    def predict(self, t: Sequence[float]) -> Sequence[float]:
        """
        Predicts the market share fraction of the new technology.

        Args:
            t: A sequence of time points.

        Returns:
            A sequence of predicted market share fractions (between 0 and 1).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        from scipy.integrate import solve_ivp

        t_arr = B.array(t)
        y0 = 1 / (1 + B.exp(-self._params['alpha'] * (t_arr[0] - self._params['t0'])))

        fun = lambda t, y: self.differential_equation(y, t, **self._params)

        sol = solve_ivp(
            fun,
            (t_arr[0], t_arr[-1]),
            [y0],
            t_eval=t_arr,
            method='LSODA',
        )
        return sol.y.flatten()

    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        """
        Calculates the R^2 score for the model fit.
        """
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
        """
        Predicts the rate of change of market share.

        This is the derivative of the logistic function, representing the
        speed of substitution.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        t_arr = B.array(t)
        y_pred = self.predict(t_arr)
        return self.differential_equation(y_pred, t_arr, **self._params)

    def fit(self, fitter, t: Sequence[float], y: Sequence[float], **kwargs):
        """
        Fits the Fisher-Pry model to the data.

        Note: The input `y` for the Fisher-Pry model should be the market
        share fraction (between 0 and 1) of the new technology.
        """
        return super().fit(fitter, t, y, **kwargs)