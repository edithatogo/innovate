from innovate.base.base import DiffusionModel, Self
from innovate.dynamics.growth.skewed import SkewedGrowth
from typing import Sequence, Dict
import numpy as np

class GompertzModel(DiffusionModel):
    """
    Implementation of the Gompertz Diffusion Model.
    This is a wrapper around the SkewedGrowth dynamics model.
    """

    def __init__(self, covariates: Sequence[str] = None):
        self._params: Dict[str, float] = {}
        self.covariates = covariates if covariates else []
        self.growth_model = SkewedGrowth()

    @property
    def param_names(self) -> Sequence[str]:
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

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None, t_eval: Sequence[float] = None) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        y0 = 1e-6

        from scipy.integrate import solve_ivp
        params = [self._params[name] for name in self.param_names]
        if t_eval is None:
            t_eval = t
        fun = lambda t, y: self.differential_equation(t, y, params, covariates, t_eval=t_eval)
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
        """The differential equation for the Gompertz model."""
        a_base = params[0]
        b_base = params[1]
        c_base = params[2]

        a_t = a_base
        b_t = b_base
        c_t = c_base
        
        if covariates:
            param_idx = 3
            for cov_name, cov_values in covariates.items():
                t = np.array(t)
                if t.ndim == 0:
                    t = np.array([t])

                t_eval = np.array(t_eval)
                if t_eval.ndim == 0:
                    t_eval = np.array([t_eval])

                cov_val_t = np.interp(t, t_eval, cov_values)
                
                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx+1] * cov_val_t
                c_t += params[param_idx+2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(y, a_t, t=t, shape_b=b_t, shape_c=c_t)

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
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
