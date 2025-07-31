from innovate.base.base import DiffusionModel
from innovate import backend
from innovate.dynamics.growth.symmetric import SymmetricGrowth
from typing import Sequence, Dict
import numpy as np


class LogisticModel(DiffusionModel):
    """
    Implementation of the Logistic Diffusion Model.
    This is a wrapper around the SymmetricGrowth dynamics model.
    """

    def __init__(self, covariates: Sequence[str] = None, t_event: float = None):
        """
        Initialize a LogisticModel with optional covariates and an internal SymmetricGrowth dynamics model.

        Parameters:
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list.
            t_event (float, optional): The time of a structural break or event.
        """
        self._params: Dict[str, float] = {}
        self.covariates = covariates if covariates else []
        self.t_event = t_event
        self.growth_model = SymmetricGrowth()

    @property
    def param_names(self) -> Sequence[str]:
        """
        Return the list of parameter names for the logistic model, including base parameters and covariate-specific coefficients.

        Returns:
            names (Sequence[str]): List of parameter names, with covariate effects prefixed by 'beta_L_', 'beta_k_', and 'beta_x0_' for each covariate.
        """
        names = ["L", "k", "x0"]
        if self.t_event is not None:
            names.extend(["L_post", "k_post", "x0_post"])
        for cov in self.covariates:
            names.extend([f"beta_L_{cov}", f"beta_k_{cov}", f"beta_x0_{cov}"])
        return names

    def initial_guesses(
        self, t: Sequence[float], y: Sequence[float]
    ) -> Dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                }
            )
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
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                }
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def predict(
        self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None
    ) -> Sequence[float]:
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

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (
                    1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0))
                )

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (
                    1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0))
                )

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: Dict[str, Sequence[float]] = None,
    ) -> float:
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
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value

    def predict_adoption_rate(
        self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def cumulative_adoption(
        self, t: Sequence[float], *params, **param_kwargs
    ) -> Sequence[float]:
        if param_kwargs:
            self.params_ = param_kwargs
        else:
            self.params_ = dict(zip(self.param_names, params))
        return self.predict(t)

    def differential_equation(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)
