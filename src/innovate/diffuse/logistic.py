from innovate.base.base import DiffusionModel, Self
from innovate.backend import current_backend as B
from innovate.dynamics.growth.symmetric import SymmetricGrowth
from typing import Sequence, Dict
import numpy as np

class LogisticModel(DiffusionModel):
    """
    Implementation of the Logistic Diffusion Model.
    This is a wrapper around the SymmetricGrowth dynamics model.
    """

    def __init__(self, covariates: Sequence[str] = None):
        """
        Initialize a LogisticModel with optional covariates and an internal SymmetricGrowth dynamics model.
        
        Parameters:
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list.
        """
        self._params: Dict[str, float] = {}
        self.covariates = covariates if covariates else []
        self.growth_model = SymmetricGrowth()

    @property
    def param_names(self) -> Sequence[str]:
        """
        Return the list of parameter names for the logistic model, including base parameters and covariate-specific coefficients.
        
        Returns:
            names (Sequence[str]): List of parameter names, with covariate effects prefixed by 'beta_L_', 'beta_k_', and 'beta_x0_' for each covariate.
        """
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
        """
        Return parameter bounds for the logistic model, including covariate effects.
        
        Parameters:
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.
        
        Returns:
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
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
        """
        Predicts the cumulative values of the logistic diffusion process at specified time points.
        
        Parameters:
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.
        
        Returns:
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.
        
        Raises:
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        y0 = 1e-6
        
        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp
        params = [self._params[name] for name in self.param_names]
        fun = lambda t, y: self.differential_equation(t, y, params, covariates, t)
        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            [y0],
            t_eval=t,
            method='LSODA',
            dense_output=True,
        )
        return sol.sol(t).flatten()



    def differential_equation(self, t, y, params, covariates, t_eval):
        """
        Defines the time derivative for the logistic growth model, incorporating covariate effects into the carrying capacity and growth rate.
        
        Parameters:
            t (float): Current time point.
            y (float): Current state value.
            params (Sequence[float]): Model parameters, including base logistic parameters and covariate coefficients.
            covariates (dict): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.
        
        Returns:
            float: The computed growth rate at time t, adjusted for covariate effects.
        """
        L_base = params[0]
        k_base = params[1]

        L_t = L_base
        k_t = k_base
        
        if covariates:
            param_idx = 3
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                
                L_t += params[param_idx] * cov_val_t
                k_t += params[param_idx+1] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(y, L_t, growth_rate=k_t)

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
        """
        Compute the coefficient of determination (R²) between observed values and model predictions.
        
        Parameters:
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.
        
        Returns:
            float: The R² score indicating the proportion of variance explained by the model predictions.
        
        Raises:
            RuntimeError: If the model has not been fitted.
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