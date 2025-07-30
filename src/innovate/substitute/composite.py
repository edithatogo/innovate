# src/innovate/substitute/composite.py

from innovate.base.base import DiffusionModel
from typing import Sequence, Dict, List
import numpy as np
import pytensor.tensor as pt
from innovate.backend import current_backend as B

class CompositeDiffusionModel(DiffusionModel):
    """
    A generic model for the diffusion of multiple, potentially interacting products.
    This model is composed of multiple individual diffusion models and an interaction matrix
    that defines how the adoption of one product affects the adoption of others.
    """

    def __init__(self, models: List[DiffusionModel], alpha: np.ndarray = None):
        """
        Initializes the CompositeDiffusionModel.
        Parameters:
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: Dict[str, float] = {}
        
        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError("Interaction matrix alpha must have shape (n_models, n_models).")
            self.alpha = alpha

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        for i, model in enumerate(self.models):
            for param_name in model.param_names:
                names.append(f"{param_name}_{i+1}")
        
        # Add interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    names.append(f"alpha_{i+1}_{j+1}")
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i+1}"] = value
        
        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i+1}_{j+1}"] = 0.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i+1}"] = value
        
        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i+1}_{j+1}"] = (-np.inf, np.inf)
        return bounds

    def predict(self, t: Sequence[float]) -> Sequence[float]:
        """
        Predicts the cumulative adoption for each product.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp
        
        # Compile the differential equation if the parameters are pytensor variables
        if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
            t_sym = pt.scalar('t')
            y_sym = pt.vector('y')
            params_sym = [pt.scalar(name) for name in self.param_names]
            
            dydt = self.differential_equation(t_sym, y_sym, params_sym)
            
            fun = pt.function(
                [t_sym, y_sym] + params_sym,
                dydt,
                on_unused_input='ignore'
            )
            
            param_values = [self._params[name] for name in self.param_names]
            fun = lambda t, y: fun(t, y, *param_values)
        else:
            fun = lambda t, y: self.differential_equation(t, y, self._params)

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method='BDF',
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def differential_equation(self, t, y, params):
        """
        Defines the composite diffusion model's differential equations.
        """
        
        is_pytensor = any(isinstance(p, pt.TensorVariable) for p in (params.values() if isinstance(params, dict) else params))

        if is_pytensor:
            dydt = pt.zeros_like(y)
            param_list = params if isinstance(params, list) else [params[name] for name in self.param_names]
        else:
            dydt = B.zeros_like(y)
            param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params


        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx:param_idx + num_params])
            param_idx += num_params
            
        alpha_params = param_list[param_idx:]
        
        if is_pytensor:
            alpha = pt.zeros((self.n_models, self.n_models))
        else:
            alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    if is_pytensor:
                        alpha = pt.set_subtensor(alpha[i, j], alpha_params[alpha_idx])
                    else:
                        alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]
            
            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(t, pt.as_tensor_variable(y[i:i+1]), model_params, None, t, is_pytensor=is_pytensor)
            
            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)
            
            if is_pytensor:
                dydt = pt.set_subtensor(dydt[i], growth_rate - interaction_effect)
            else:
                dydt[i] = growth_rate - interaction_effect

        return dydt

    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y, axis=0))**2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        y_pred = self.predict(t)
        rates = np.array([self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, y_pred)])
        return rates
