from innovate.base.base import DiffusionModel, Self
from innovate.backend import current_backend as B
from innovate.dynamics.growth.dual_influence import DualInfluenceGrowth
from typing import Sequence, Dict
import numpy as np

class BassModel(DiffusionModel):
    """
    Implementation of the Bass Diffusion Model.
    This is a wrapper around the DualInfluenceGrowth dynamics model.
    """

    def __init__(self, covariates: Sequence[str] = None):
        self._params: Dict[str, float] = {}
        self.covariates = covariates if covariates else []
        self.growth_model = DualInfluenceGrowth()

    @property
    def param_names(self) -> Sequence[str]:
        names = ["p", "q", "m"]
        for cov in self.covariates:
            names.extend([f"beta_p_{cov}", f"beta_q_{cov}", f"beta_m_{cov}"])
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y) * 1.1,
        }
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y), np.inf),
        }
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
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
        )
        return sol.y.flatten()

    def differential_equation(self, t, y, params, covariates, t_eval):
        """The differential equation for the Bass model."""
        p_base = params[0]
        q_base = params[1]
        m_base = params[2]

        p_t = p_base
        q_t = q_base
        m_t = m_base
        
        if covariates:
            param_idx = 3
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                
                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx+1] * cov_val_t
                m_t += params[param_idx+2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(y, m_t, innovation_coeff=p_t, imitation_coeff=q_t)

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
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
