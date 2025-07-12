from ..base import DiffusionModel
import numpy as np
from typing import Sequence, Dict

class NortonBassModel(DiffusionModel):
    """
    Norton-Bass Model for successive generations of technologies.
    """
    def __init__(self, n_generations: int = 1, covariates: Sequence[str] = None):
        if n_generations < 1:
            raise ValueError("Number of generations must be at least 1.")
        self.n_generations = n_generations
        self._params: Dict[str, float] = {}
        self.covariates = covariates if covariates else []

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        for i in range(self.n_generations):
            names.extend([f"p{i+1}", f"q{i+1}", f"m{i+1}"])
        
        for cov in self.covariates:
            for i in range(self.n_generations):
                names.extend([f"beta_p{i+1}_{cov}", f"beta_q{i+1}_{cov}", f"beta_m{i+1}_{cov}"])
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {}
        max_y = np.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i+1}"] = 0.001
            guesses[f"q{i+1}"] = 0.1
            guesses[f"m{i+1}"] = max_y / self.n_generations
        
        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i+1}_{cov}"] = 0.0
                guesses[f"beta_q{i+1}_{cov}"] = 0.0
                guesses[f"beta_m{i+1}_{cov}"] = 0.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bounds = {}
        max_y = np.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i+1}"] = (1e-6, 0.1)
            bounds[f"q{i+1}"] = (1e-6, 1.0)
            bounds[f"m{i+1}"] = (0, max_y * 2)
            
        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i+1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i+1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i+1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        from scipy.integrate import solve_ivp
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(self.n_generations)
        
        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]
        
        fun = lambda t, y: self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method='LSODA',
        )
        return sol.y.T

    def differential_equation(self, t, y, params, covariates, t_eval):
        """
        System of differential equations for the Norton-Bass model.
        """
        
        p_base = params[:self.n_generations]
        q_base = params[self.n_generations:2*self.n_generations]
        m_base = params[2*self.n_generations:3*self.n_generations]

        p_t = np.array(p_base)
        q_t = np.array(q_base)
        m_t = np.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx+1] * cov_val_t
                    m_t[i] += params[param_idx+2] * cov_val_t
                    param_idx += 3
        
        dydt = np.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = np.ravel(y)
                cannibalization = np.sum(y_flat[i+1:])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        
        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict_adoption_rate(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]
        
        rates = np.array([self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)])
        return rates