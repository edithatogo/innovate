
import numpy as np
from ..base import DiffusionModel
from typing import Sequence, Dict

class MultiProductDiffusionModel(DiffusionModel):
    """
    A generalized model for the diffusion of multiple competing products.
    """
    def __init__(self, n_products: int):
        if n_products < 1:
            raise ValueError("Number of products must be at least 1.")
        self.n_products = n_products
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        for i in range(self.n_products):
            names.extend([f"p{i+1}", f"q{i+1}", f"m{i+1}"])
        
        for i in range(self.n_products):
            for j in range(self.n_products):
                if i != j:
                    names.append(f"alpha_{i+1}_{j+1}")
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {}
        max_y = np.max(y)
        for i in range(self.n_products):
            guesses[f"p{i+1}"] = 0.001
            guesses[f"q{i+1}"] = 0.1
            guesses[f"m{i+1}"] = max_y / self.n_products

        for i in range(self.n_products):
            for j in range(self.n_products):
                if i != j:
                    guesses[f"alpha_{i+1}_{j+1}"] = 1.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bounds = {}
        max_y = np.max(y)
        for i in range(self.n_products):
            bounds[f"p{i+1}"] = (1e-6, 0.1)
            bounds[f"q{i+1}"] = (1e-6, 1.0)
            bounds[f"m{i+1}"] = (0, max_y * 2)

        for i in range(self.n_products):
            for j in range(self.n_products):
                if i != j:
                    bounds[f"alpha_{i+1}_{j+1}"] = (0, 2.0)
        return bounds

    def predict(self, t: Sequence[float]) -> Sequence[float]:
        from scipy.integrate import solve_ivp
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(self.n_products)
        y0[0] = 1e-6 # Start with a small value for the first product

        params = [self._params[name] for name in self.param_names]
        
        sol = solve_ivp(
            self.differential_equation,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            args=tuple(params),
            method='LSODA',
        )
        return sol.y.T

    def differential_equation(self, t, y, *params):
        
        num_products = self.n_products
        p = params[:num_products]
        q = params[num_products:2*num_products]
        m = params[2*num_products:3*num_products]
        
        alpha_params = params[3*num_products:]
        alpha = np.zeros((num_products, num_products))
        alpha_idx = 0
        for i in range(num_products):
            for j in range(num_products):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        dydt = np.zeros_like(y)
        for i in range(num_products):
            interaction_term = sum(alpha[i, j] * y[j] for j in range(num_products) if i != j)
            dydt[i] = (p[i] + q[i] * y[i] / m[i]) * (m[i] - y[i] - interaction_term) if m[i] > 0 else 0

        return dydt

    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        y_pred = self.predict(t)
        params = [self._params[name] for name in self.param_names]
        
        rates = np.array([self.differential_equation(ti, yi, *params) for ti, yi in zip(t, y_pred)])
        return rates

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value
