from innovate.backend import current_backend as B
import numpy as np
from ..base import DiffusionModel
from typing import Sequence, Dict

class MultiProductDiffusionModel(DiffusionModel):
    """
    A generalized model for the diffusion of multiple competing products.
    """
    def __init__(self, n_products: int, p: Sequence[float] = None, Q: Sequence[Sequence[float]] = None, m: Sequence[float] = None, names: Sequence[str] = None, covariates: Sequence[str] = None):
        if n_products < 1:
            raise ValueError("Number of products must be at least 1.")
        self.n_products = n_products
        self._params: Dict[str, float] = {}
        self.covariates = covariates if covariates else []

        self.p = B.array(p) if p is not None else None
        self.Q = B.array(Q) if Q is not None else None
        self.m = B.array(m) if m is not None else None
        self.names = names

        if self.p is not None and self.Q is not None and self.m is not None:
            if not (len(self.p) == self.n_products and 
                    self.Q.shape == (self.n_products, self.n_products) and 
                    len(self.m) == self.n_products):
                raise ValueError("Dimensions of p, Q, and m must be consistent with n_products.")
        
        if self.names is not None and len(self.names) != self.n_products:
            raise ValueError("Number of names must match n_products.")

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        # Add p, q, m parameters for each product
        for prefix in ["p", "q", "m"]:
            for i in range(self.n_products):
                names.append(f"{prefix}{i+1}")
        
        # Add alpha (interaction) parameters
        for i in range(self.n_products):
            for j in range(self.n_products):
                if i != j:
                    names.append(f"alpha_{i+1}_{j+1}")

        # Add covariate-related beta parameters
        for cov in self.covariates:
            # Betas for p, q, m
            for prefix in ["p", "q", "m"]:
                for i in range(self.n_products):
                    names.append(f"beta_{prefix}{i+1}_{cov}")
            # Betas for alpha
            for i in range(self.n_products):
                for j in range(self.n_products):
                    if i != j:
                        names.append(f"beta_alpha_{i+1}_{j+1}_{cov}")
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        
        # Initial guesses for p, q, m
        for i in range(self.n_products):
            guesses[f"p{i+1}"] = 0.001
        for i in range(self.n_products):
            guesses[f"q{i+1}"] = 0.1
        for i in range(self.n_products):
            guesses[f"m{i+1}"] = max_y / self.n_products

        # Initial guesses for alpha
        for i in range(self.n_products):
            for j in range(self.n_products):
                if i != j:
                    guesses[f"alpha_{i+1}_{j+1}"] = 1.0

        # Initial guesses for betas
        for cov in self.covariates:
            for prefix in ["p", "q", "m"]:
                for i in range(self.n_products):
                    guesses[f"beta_{prefix}{i+1}_{cov}"] = 0.0
            for i in range(self.n_products):
                for j in range(self.n_products):
                    if i != j:
                        guesses[f"beta_alpha_{i+1}_{j+1}_{cov}"] = 0.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        
        # Bounds for p, q, m
        for i in range(self.n_products):
            bounds[f"p{i+1}"] = (1e-6, 0.1)
        for i in range(self.n_products):
            bounds[f"q{i+1}"] = (1e-6, 1.0)
        for i in range(self.n_products):
            bounds[f"m{i+1}"] = (0, max_y * 2)

        # Bounds for alpha
        for i in range(self.n_products):
            for j in range(self.n_products):
                if i != j:
                    bounds[f"alpha_{i+1}_{j+1}"] = (0, 2.0)

        # Bounds for betas
        for cov in self.covariates:
            for prefix in ["p", "q", "m"]:
                for i in range(self.n_products):
                    bounds[f"beta_{prefix}{i+1}_{cov}"] = (-np.inf, np.inf)
            for i in range(self.n_products):
                for j in range(self.n_products):
                    if i != j:
                        bounds[f"beta_alpha_{i+1}_{j+1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        from scipy.integrate import solve_ivp
        
        t_arr = B.array(t)
        
        y0 = B.zeros(self.n_products)
        y0[0] = 1e-6

        if self.p is not None and self.Q is not None and self.m is not None:
            # Use pre-defined parameters if available (for direct simulation)
            p_vals = self.p
            q_vals = self.Q.diagonal()
            m_vals = self.m
            
            alpha_flat = []
            for i in range(self.n_products):
                for j in range(self.n_products):
                    if i != j:
                        alpha_flat.append(self.Q[i, j])

            params_for_ode = list(p_vals) + list(q_vals) + list(m_vals) + alpha_flat
            
            if self.covariates and self._params:
                for cov in self.covariates:
                    for i in range(self.n_products):
                        params_for_ode.append(self._params.get(f"beta_p{i+1}_{cov}", 0.0))
                        params_for_ode.append(self._params.get(f"beta_q{i+1}_{cov}", 0.0))
                        params_for_ode.append(self._params.get(f"beta_m{i+1}_{cov}", 0.0))
                    for i in range(self.n_products):
                        for j in range(self.n_products):
                            if i != j:
                                params_for_ode.append(self._params.get(f"beta_alpha_{i+1}_{j+1}_{cov}", 0.0))

        elif self._params:
            # Use fitted parameters if available
            params_for_ode = [self._params[name] for name in self.param_names]
        else:
            raise RuntimeError("Model parameters (p, Q, m) are not set, and model has not been fitted yet. Call .fit() or set parameters directly.")
        
        fun = lambda t_val, y_val: self.differential_equation(t_val, y_val, params_for_ode, covariates, t_arr)

        sol = solve_ivp(
            fun,
            (t_arr[0], t_arr[-1]),
            y0,
            t_eval=t_arr,
            method='LSODA',
        )
        return sol.y.T

    def differential_equation(self, t, y, params, covariates, t_eval):
        # Unpack the params_tuple
        all_params_flat = params
        n_products = self.n_products
        covariates_dict = covariates
        covariate_names = self.covariates
        
        # Calculate the number of alpha parameters (off-diagonal elements in Q)
        num_alpha_params = n_products * (n_products - 1)

        # Extract base parameters
        p_base = B.array(all_params_flat[:n_products])
        q_base = B.array(all_params_flat[n_products:2*n_products])
        m_base = B.array(all_params_flat[2*n_products:3*n_products])
        
        alpha_base_flat = B.array(all_params_flat[3*n_products : 3*n_products + num_alpha_params])
        
        # Initialize time-varying parameters with base values
        p_t = np.copy(p_base)
        q_t = np.copy(q_base)
        m_t = np.copy(m_base)
        alpha_t_flat = np.copy(alpha_base_flat)

        # Apply covariate effects
        if covariates_dict:
            # The offset for beta coefficients starts after all base p, q, m, and alpha parameters
            param_idx_offset = 3 * n_products + num_alpha_params 
            
            for cov_name in covariate_names:
                cov_values = covariates_dict[cov_name]
                cov_val_t = np.interp(t, t_eval, cov_values)
                
                # Add covariate effects to p, q, m
                for i in range(n_products):
                    p_t[i] += all_params_flat[param_idx_offset + i * 3] * cov_val_t
                    q_t[i] += all_params_flat[param_idx_offset + i * 3 + 1] * cov_val_t
                    m_t[i] += all_params_flat[param_idx_offset + i * 3 + 2] * cov_val_t
                
                # Add covariate effects to alpha
                # The alpha betas follow the m betas for each product
                alpha_beta_start_idx_for_cov = param_idx_offset + n_products * 3
                current_alpha_beta_idx = 0
                for i in range(n_products):
                    for j in range(n_products):
                        if i != j:
                            alpha_t_flat[current_alpha_beta_idx] += all_params_flat[alpha_beta_start_idx_for_cov + current_alpha_beta_idx] * cov_val_t
                            current_alpha_beta_idx += 1
                param_idx_offset = alpha_beta_start_idx_for_cov + num_alpha_params # Update offset for next covariate

        # Reshape alpha_t_flat back to matrix
        alpha_t = B.zeros((n_products, n_products))
        alpha_idx = 0
        for i in range(n_products):
            for j in range(n_products):
                if i != j:
                    alpha_t[i, j] = alpha_t_flat[alpha_idx]
                    alpha_idx += 1

        dydt = B.zeros_like(y)
        for i in range(n_products):
            interaction_term = sum(alpha_t[i, j] * y[j] for j in range(n_products) if i != j)
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - interaction_term) if m_t[i] > 0 else 0

        return dydt

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict_adoption_rate(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]
        
        rates = B.array([self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)])
        return rates

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value
