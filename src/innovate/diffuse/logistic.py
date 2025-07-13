from innovate.base.base import DiffusionModel, Self
from innovate.backend import current_backend as B
from typing import Sequence, Dict
import numpy as np

class LogisticModel(DiffusionModel):
    """Implementation of the Logistic Diffusion Model."""

    def __init__(self, covariates: Sequence[str] = None):
        self._params: Dict[str, float] = {}
        self.covariates = covariates if covariates else []

    @property
    def param_names(self) -> Sequence[str]:
        names = ["L", "k", "x0"]
        for cov in self.covariates:
            names.extend([f"beta_L_{cov}", f"beta_k_{cov}", f"beta_x0_{cov}"])
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        from scipy.integrate import solve_ivp
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        t_arr = B.array(t)
        
        y0 = np.zeros(1)
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]
        
        fun = lambda t, y: self.differential_equation(t, y, params, covariates, t_arr)

        sol = solve_ivp(
            fun,
            (t_arr[0], t_arr[-1]),
            y0,
            t_eval=t_arr,
            method='LSODA',
        )
        return sol.y.flatten()

    def differential_equation(self, t, y, params, covariates, t_eval):
        """
        Defines the logistic differential equation with optional covariate effects.
        
        At each time point `t`, computes the rate of change for the logistic model, adjusting the parameters `L`, `k`, and `x0` by the influence of covariates if provided. Returns zero if the carrying capacity parameter `L_t` is not positive.

        Parameters:
            t (float): Current time.
            y (array-like): Current value(s) of the dependent variable.
            params (array-like): Model parameters, including base and covariate coefficients.
            covariates (dict): Optional mapping of covariate names to their values over time.
            t_eval (array-like): Time points corresponding to covariate values.
        
        Returns:
            float: The computed derivative at time t, or zero if the carrying capacity is non-positive.
        """
        
        L_base = params[0]
        k_base = params[1]
        x0_base = params[2]

        L_t = L_base
        k_t = k_base
        x0_t = x0_base
        
        if covariates:
            param_idx = 3
            for cov_name, cov_values in covariates.items():
                # Interpolate covariate values at time t
                cov_val_t = np.interp(t, t_eval, cov_values)
                
                L_t += params[param_idx] * cov_val_t
                k_t += params[param_idx+1] * cov_val_t
                x0_t += params[param_idx+2] * cov_val_t
                param_idx += 3

        return B.switch(B.gt(L_t, 0), k_t * y[0] * (1 - y[0] / L_t), 0)

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
        """

        Compute the coefficient of determination (R²) between observed values and model predictions.
        
        Parameters:
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points in `t`.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.
        
        Returns:
            float: The R² score indicating the proportion of variance in `y` explained by the model predictions.
        
        Raises:
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value

    def predict_adoption_rate(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]
        
        rates = np.array([self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)])
        return rates
