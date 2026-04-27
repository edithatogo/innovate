from collections.abc import Sequence

import numpy as np

from innovate.backend import current_backend as B
from innovate.base.base import DiffusionModel
from innovate.dynamics.growth.skewed import SkewedGrowth


class GompertzModel(DiffusionModel):
    """Gompertz diffusion model backed by ``SkewedGrowth``."""

    def __init__(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the model with optional covariates and event timing."""
        self._params: dict[str, float] = {}
        self.covariates = covariates or []
        self.t_event = t_event
        self.growth_model = SkewedGrowth()

    @property
    def param_names(self) -> Sequence[str]:
        """Return the fitted parameter names."""
        names = ["a", "b", "c"]
        if self.t_event is not None:
            names.extend(["a_post", "b_post", "c_post"])
        for cov in self.covariates:
            names.extend([f"beta_a_{cov}", f"beta_b_{cov}", f"beta_c_{cov}"])
        return names

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the observed data."""
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predict cumulative adoption for the provided time grid."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def differential_equation(self, t, y, params, covariates, t_eval):
        """Return the instantaneous growth rate at ``t``."""
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Return the coefficient of determination for the fitted model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = value

    def predict_adoption_rate(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def cumulative_adoption(self, t: Sequence[float], *params) -> Sequence[float]:
        self.params_ = dict(zip(self.param_names, params))
        return self.predict(t)
