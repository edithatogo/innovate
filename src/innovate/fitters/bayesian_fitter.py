# src/innovate/fitters/bayesian_fitter.py

import pymc as pm
import numpy as np
import pytensor.tensor as pt
from typing import Sequence, Dict
from innovate.base.base import DiffusionModel

class BayesianFitter:
    """
    A fitter that uses Bayesian methods (specifically, PyMC) to estimate
    model parameters.
    """

    def __init__(self, model: DiffusionModel, draws: int = 2000, tune: int = 1000, chains: int = 4):
        self.model = model
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.trace = None

    def fit(self, t: Sequence[float], y: np.ndarray, **kwargs):
        """
        Fits the model to the data using PyMC.
        """
        
        ode_func = self.model.differential_equation

        with pm.Model() as bayesian_model:
            # Define priors for the model parameters
            priors = self._define_priors(t, y)
            
            # Convert the dictionary of priors to a list for the ODE function
            param_list = [priors[name] for name in self.model.param_names]

            # ODE solver
            ode_solution = pm.ode.DifferentialEquation(
                func=ode_func,
                times=t,
                n_states=1,
                n_theta=len(param_list),
                t0=0,
            )
            
            # The expected value of the data, given the parameters
            mu = ode_solution(y0=[y[0]], theta=param_list)

            # Likelihood of the data
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            likelihood = pm.Normal("likelihood", mu=mu[:, 0], sigma=sigma, observed=y)

            # Sample from the posterior
            self.trace = pm.sample(self.draws, tune=self.tune, chains=self.chains, **kwargs)

        # Set the model parameters to the mean of the posterior
        self.model.params_ = self.get_parameter_estimates()
        return self

    def _define_priors(self, t: Sequence[float], y: np.ndarray) -> Dict[str, pm.Distribution]:
        """
        Defines the priors for the model parameters.
        """
        priors = {}
        initial_guesses = self.model.initial_guesses(t, y)
        
        if self.model.__class__.__name__ == 'LogisticModel':
            priors['L'] = pm.HalfNormal('L', sigma=initial_guesses['L'])
            priors['k'] = pm.HalfNormal('k', sigma=initial_guesses['k'])
            priors['x0'] = pm.Normal('x0', mu=initial_guesses['x0'], sigma=5.0)
        else:
            bounds = self.model.bounds(t, y)
            for param_name in self.model.param_names:
                lower, upper = bounds[param_name]
                if lower is None or not np.isfinite(lower):
                    lower = -np.inf
                if upper is None or not np.isfinite(upper):
                    upper = np.inf
                
                if np.isinf(lower) and np.isinf(upper):
                    priors[param_name] = pm.Normal(param_name, mu=initial_guesses[param_name], sigma=1.0)
                elif np.isinf(upper):
                    priors[param_name] = pm.HalfNormal(param_name, sigma=1.0)
                else:
                    priors[param_name] = pm.Uniform(param_name, lower=lower, upper=upper)
        
        return priors

    def get_parameter_estimates(self) -> Dict[str, float]:
        """
        Returns the mean of the posterior distribution for each parameter.
        """
        if self.trace is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        estimates = {}
        for param_name in self.model.param_names:
            estimates[param_name] = self.trace.posterior[param_name].mean().item()
        return estimates

    def get_confidence_intervals(self, alpha: float = 0.05) -> Dict[str, tuple]:
        """
        Returns the (1-alpha)% confidence intervals for each parameter.
        """
        if self.trace is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        intervals = {}
        for param_name in self.model.param_names:
            intervals[param_name] = tuple(pm.stats.hdi(self.trace.posterior[param_name], 1 - alpha))
        return intervals

    def get_summary(self):
        """
        Returns a summary of the posterior distribution.
        """
        if self.trace is None:
            raise RuntimeError("The model has not been fitted yet.")
        return pm.summary(self.trace)
