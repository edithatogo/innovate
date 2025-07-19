from innovate.base.base import DiffusionModel, Self
from innovate.backend import current_backend as B
from innovate.dynamics.growth.skewed import SkewedGrowth
from typing import Sequence, Dict
import numpy as np

class GompertzModel(DiffusionModel):
    """
    Implementation of the Gompertz Diffusion Model.
    This is a wrapper around the SkewedGrowth dynamics model.
    """

    def __init__(self, covariates: Sequence[str] = None):
        """
        Initialize a Gompertz diffusion model with optional covariates.
        
        Creates an empty parameter dictionary, stores the provided covariate names, and instantiates a SkewedGrowth dynamics model for growth rate computation.
        """
        self._params: Dict[str, float] = {}
        self.covariates = covariates if covariates else []
        self.growth_model = SkewedGrowth()

    @property
    def param_names(self) -> Sequence[str]:
        """
        Return the list of model parameter names, including base parameters and covariate-specific coefficients.
        
        Returns:
            Sequence[str]: List of parameter names for the model, with additional parameters for each covariate in the form 'beta_a_{cov}', 'beta_b_{cov}', and 'beta_c_{cov}'.
        """
        names = ["a", "b", "c"]
        for cov in self.covariates:
            names.extend([f"beta_a_{cov}", f"beta_b_{cov}", f"beta_c_{cov}"])
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        """
        Return parameter bounds for the Gompertz model based on observed data and covariates.
        
        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.
        
        Parameters:
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.
        
        Returns:
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        """
        Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.
        
        Parameters:
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.
        
        Returns:
            Sequence[float]: Predicted cumulative adoption values at each time point.
        
        Raises:
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp
        params = [self._params[name] for name in self.param_names]
        fun = lambda t, y: self.differential_equation(t, y, params, covariates, t)
        y0 = [1e-6]
        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method='LSODA',
            dense_output=True,
        )
        return sol.sol(t).flatten()

    def differential_equation(self, t, y, params, covariates, t_eval):
        """
        Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.
        
        Parameters:
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.
        
        Returns:
            float: The instantaneous growth rate at time t.
        """
        a_base = params[0]
        b_base = params[1]
        c_base = params[2]

        a_t = a_base
        b_t = b_base
        c_t = c_base
        
        if covariates:
            param_idx = 3
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                
                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx+1] * cov_val_t
                c_t += params[param_idx+2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(y, a_t, t=t, shape_b=b_t, shape_c=c_t)

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
        """
        Compute the coefficient of determination (R²) between observed data and model predictions.
        
        Parameters:
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.
        
        Returns:
            float: R² score indicating the proportion of variance explained by the model predictions.
        
        Raises:
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = np.sum((np.array(y) - y_pred) ** 2)
        ss_tot = np.sum((np.array(y) - np.mean(np.array(y))) ** 2)
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