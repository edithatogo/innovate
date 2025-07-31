# minimal_repro.py

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from typing import Sequence, Dict


class BassModel:
    """A minimal implementation of the Bass Diffusion Model."""

    @property
    def param_names(self) -> Sequence[str]:
        return ["p", "q", "m"]

    def initial_guesses(
        self, t: Sequence[float], y: Sequence[float]
    ) -> Dict[str, float]:
        return {"p": 0.001, "q": 0.1, "m": np.max(y) * 1.1}

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        return {"p": (1e-6, 0.1), "q": (1e-6, 1.0), "m": (np.max(y), np.inf)}

    def differential_equation(self, t, y, params, covariates, t_eval):
        p_t, q_t, m_t = params[0], params[1], params[2]
        return pt.switch(m_t > 0, (p_t + q_t * (y[0] / m_t)) * (m_t - y[0]), 0)


class BayesianFitter:
    """A minimal implementation of the BayesianFitter."""

    def __init__(
        self, model: BassModel, draws: int = 2000, tune: int = 1000, chains: int = 1
    ):
        self.model = model
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.trace = None

    def fit(self, t: Sequence[float], y: np.ndarray, **kwargs):
        ode_func = self.model.differential_equation

        def ode_func_wrapper(y, t, p):
            return ode_func(t, y, p, covariates=None, t_eval=t)

        with pm.Model():
            priors = self._define_priors(t, y)
            param_list = [priors[name] for name in self.model.param_names]

            ode_solution = pm.ode.DifferentialEquation(
                func=ode_func_wrapper,
                times=t,
                n_states=1,
                n_theta=len(param_list),
                t0=0,
            )

            mu = ode_solution(y0=[y[0]], theta=param_list)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            pm.Normal("likelihood", mu=mu[:, 0], sigma=sigma, observed=y)

            self.trace = pm.sample(
                self.draws, tune=self.tune, chains=self.chains, **kwargs
            )

        return self

    def _define_priors(
        self, t: Sequence[float], y: np.ndarray
    ) -> Dict[str, pm.Distribution]:
        priors = {}
        initial_guesses = self.model.initial_guesses(t, y)
        bounds = self.model.bounds(t, y)
        for param_name in self.model.param_names:
            lower, upper = bounds[param_name]
            if lower is None or not np.isfinite(lower):
                lower = -np.inf
            if upper is None or not np.isfinite(upper):
                upper = np.inf

            if np.isinf(lower) and np.isinf(upper):
                priors[param_name] = pm.Normal(
                    param_name, mu=initial_guesses[param_name], sigma=1.0
                )
            elif np.isinf(upper):
                priors[param_name] = pm.HalfNormal(param_name, sigma=1.0)
            else:
                priors[param_name] = pm.Uniform(param_name, lower=lower, upper=upper)

        return priors


if __name__ == "__main__":
    t = np.linspace(0, 20, 100)
    p, q, m = 0.03, 0.38, 1.0
    y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
    y += np.random.normal(0, 0.01, len(t))

    model = BassModel()
    fitter = BayesianFitter(model, draws=1000, tune=1000, chains=1)
    fitter.fit(t, y)
    print("Fit completed successfully.")
