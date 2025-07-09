# src/innovate/compete/lotka_volterra.py

from innovate.base.base import DiffusionModel
from innovate.backend import current_backend as B
from typing import Sequence, Dict
import numpy as np

class LotkaVolterraModel(DiffusionModel):
    """
    Implementation of the Lotka-Volterra model for competitive diffusion.

    This model describes the interaction between two competing technologies or
    products, where the adoption of each is influenced by the other.
    """

    def __init__(self):
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        """
        Returns the names of the model parameters:
        - alpha1: Growth rate of technology 1
        - beta1: Competition parameter from technology 2 to 1
        - alpha2: Growth rate of technology 2
        - beta2: Competition parameter from technology 1 to 2
        """
        return ["alpha1", "beta1", "alpha2", "beta2"]

    def initial_guesses(self, t: Sequence[float], y: np.ndarray) -> Dict[str, float]:
        """
        Provides initial guesses for the model parameters by performing a
        linear regression on the linearized Lotka-Volterra equations.
        """
        y = np.array(y)
        t = np.array(t)
        dt = np.gradient(t)
        
        # Avoid division by zero for y1 and y2
        y1 = np.clip(y[:, 0], 1e-6, 1)
        y2 = np.clip(y[:, 1], 1e-6, 1)

        # Estimate derivatives
        dy1_dt = np.gradient(y1, dt, edge_order=2)
        dy2_dt = np.gradient(y2, dt, edge_order=2)

        # Linearize the equations:
        # dy1/dt / y1 = alpha1 - alpha1*y1 - beta1*y2
        # dy2/dt / y2 = alpha2 - alpha2*y2 - beta2*y1
        
        # Prepare for linear regression for tech 1
        X1 = np.vstack([-y1, -y2]).T
        Y1 = dy1_dt / y1 - np.mean(-y1) # Centering the response variable
        
        try:
            # Fit alpha1 and beta1
            params1, _, _, _ = np.linalg.lstsq(X1, Y1, rcond=None)
            alpha1_guess, beta1_guess = params1[0], params1[1]
        except np.linalg.LinAlgError:
            alpha1_guess, beta1_guess = 0.1, 0.01

        # Prepare for linear regression for tech 2
        X2 = np.vstack([-y2, -y1]).T
        Y2 = dy2_dt / y2 - np.mean(-y2) # Centering the response variable

        try:
            # Fit alpha2 and beta2
            params2, _, _, _ = np.linalg.lstsq(X2, Y2, rcond=None)
            alpha2_guess, beta2_guess = params2[0], params2[1]
        except np.linalg.LinAlgError:
            alpha2_guess, beta2_guess = 0.1, 0.01

        return {
            "alpha1": max(0, alpha1_guess),
            "beta1": max(0, beta1_guess),
            "alpha2": max(0, alpha2_guess),
            "beta2": max(0, beta2_guess),
        }

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        """Returns bounds for the model parameters."""
        return {
            "alpha1": (0, np.inf),
            "beta1": (0, np.inf),
            "alpha2": (0, np.inf),
            "beta2": (0, np.inf),
        }

    def predict(self, t: Sequence[float], y0: Sequence[float]) -> np.ndarray:
        """
        Predicts the market share of both technologies over time.

        This requires solving a system of ordinary differential equations (ODEs).

        Args:
            t: A sequence of time points.
            y0: The initial market shares for the two technologies [y1_0, y2_0].

        Returns:
            An array where each row corresponds to a time point and columns
            correspond to the market share of each technology.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        from scipy.integrate import odeint

        def system(y, t, alpha1, beta1, alpha2, beta2):
            y1, y2 = y
            dy1_dt = alpha1 * y1 * (1 - y1) - beta1 * y1 * y2
            dy2_dt = alpha2 * y2 * (1 - y2) - beta2 * y1 * y2
            return [dy1_dt, dy2_dt]

        solution = odeint(
            system,
            y0,
            t,
            args=(
                self._params["alpha1"],
                self._params["beta1"],
                self._params["alpha2"],
                self._params["beta2"],
            ),
        )
        return solution

    def fit(self, t: Sequence[float], y: np.ndarray, **kwargs):
        """
        Fits the Lotka-Volterra model to the data.

        This implementation uses `scipy.optimize.minimize` to find the best
        parameters by minimizing the sum of squared errors.

        Args:
            t: A sequence of time points.
            y: A 2D array of observed data, where y[:, 0] is the data for the
               first technology and y[:, 1] is for the second.
            kwargs: Additional keyword arguments for `scipy.optimize.minimize`.
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
            options={'maxiter': 10000},
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
        """
        Calculates the R^2 score for the model fit.

        Args:
            t: A sequence of time points.
            y: A 2D array of observed data.

        Returns:
            The R^2 score.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y = np.array(y)
        y0 = y[0, :]
        y_pred = self.predict(t, y0)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict_adoption_rate(self, t: Sequence[float], y0: Sequence[float]) -> np.ndarray:
        """
        Predicts the rate of change of market share for both technologies.

        Args:
            t: A sequence of time points.
            y0: The initial market shares for the two technologies [y1_0, y2_0].

        Returns:
            An array containing the adoption rates for each technology at each
            time point.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, y0)
        alpha1 = self._params["alpha1"]
        beta1 = self._params["beta1"]
        alpha2 = self._params["alpha2"]
        beta2 = self._params["beta2"]

        dy1_dt = alpha1 * y_pred[:, 0] * (1 - y_pred[:, 0]) - beta1 * y_pred[:, 0] * y_pred[:, 1]
        dy2_dt = alpha2 * y_pred[:, 1] * (1 - y_pred[:, 1]) - beta2 * y_pred[:, 0] * y_pred[:, 1]

        return np.vstack([dy1_dt, dy2_dt]).T