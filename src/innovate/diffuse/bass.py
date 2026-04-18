from collections.abc import Sequence

import numpy as np

from innovate import backend
from innovate.base.base import DiffusionModel
from innovate.dynamics.growth.dual_influence import DualInfluenceGrowth
from innovate.utils.validation import (
    validate_covariates,
    validate_covariates_dict,
    validate_float,
    validate_positive_numeric_sequence,
    validate_sequence_numeric,
    validate_time_series,
)


class BassModel(DiffusionModel):
    """Implementation of the Bass Diffusion Model.
    This is a wrapper around the DualInfluenceGrowth dynamics model.
    """

    def __init__(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float(t_event, "t_event")
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    @property
    def param_names(self) -> Sequence[str]:
        """Return the list of parameter names for the Bass model, including base parameters and covariate-related coefficients.

        Returns
        -------
            names (Sequence[str]): List of parameter names, with covariate effects included if applicable.
        """
        names = ["p", "q", "m"]
        if self.t_event is not None:
            names.extend(["p_post", "q_post", "m_post"])
        for cov in self.covariates:
            names.extend([f"beta_p_{cov}", f"beta_q_{cov}", f"beta_m_{cov}"])
        return names

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Allow singleton series here so callers can inspect defaults before fitting.
        t_arr = validate_sequence_numeric(t, "t")
        y_arr = validate_positive_numeric_sequence(y, "y")
        if len(t_arr) != len(y_arr):
            raise ValueError("Length of 't' must match length of 'y'")
        if len(t_arr) > 1 and not np.all(np.diff(t_arr) >= 0):
            raise ValueError("'t' values must be non-decreasing")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Allow singleton series here so callers can inspect parameter domains before fitting.
        t_arr = validate_sequence_numeric(t, "t")
        y_arr = validate_positive_numeric_sequence(y, "y")
        if len(t_arr) != len(y_arr):
            raise ValueError("Length of 't' must match length of 'y'")
        if len(t_arr) > 1 and not np.all(np.diff(t_arr) >= 0):
            raise ValueError("'t' values must be non-decreasing")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 0.0

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]
        return self._predict_with_backend(t_arr, y0, params, validated_covariates, required_params)

    def _predict_with_backend(
        self,
        t_arr: np.ndarray,
        y0: float,
        params: Sequence[float],
        validated_covariates: dict[str, Sequence[float]] | None,
        required_params: Sequence[str],
    ) -> np.ndarray:
        """Run the prediction step using the active backend."""
        use_jax_backend, solve_backend = self._resolve_prediction_backend(t_arr, params)
        self._validate_prediction_parameters(required_params, params, use_jax_backend)

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t, backend_impl=solve_backend)

        if use_jax_backend:
            sol = solve_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:

            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def _resolve_prediction_backend(self, t: Sequence[float], params: Sequence[float]) -> tuple[bool, object]:
        """Determine whether the prediction path needs the JAX backend."""
        try:
            from innovate.backends.jax_backend import JaxBackend
        except ImportError:  # pragma: no cover - optional dependency
            return False, backend.current_backend

        if isinstance(backend.current_backend, JaxBackend):
            return True, backend.current_backend

        try:
            import jax

            jax_types = tuple(
                candidate
                for candidate in (
                    getattr(jax, "Array", None),
                    getattr(jax.core, "Tracer", None),
                )
                if candidate is not None
            )
        except ImportError:  # pragma: no cover - jax optional
            jax_types = ()

        if jax_types and (isinstance(t, jax_types) or any(isinstance(param_val, jax_types) for param_val in params)):
            return True, JaxBackend()

        return False, backend.current_backend

    def _validate_prediction_parameters(
        self, required_params: Sequence[str], params: Sequence[float], use_jax_backend: bool
    ) -> None:
        """Validate finite parameter values for eager NumPy execution only."""
        if use_jax_backend:
            return
        for param_name, param_val in zip(required_params, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

    def differential_equation(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
        backend_impl=None,
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        backend_impl = backend_impl or backend.current_backend

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend_impl.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        if np.isscalar(m_t) and float(m_t) <= 0:
            return 0.0

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            pt = None

        if pt is not None and isinstance(
            m_t,
            pt.TensorVariable,
        ):  # pragma: no cover - depends on pytensor
            return pt.switch(m_t > 0, rate, 0.0)
        return backend_impl.where(m_t > 0, rate, 0.0)

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
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
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def cumulative_adoption(self, t: Sequence[float], *params) -> np.ndarray:
        self.params_ = dict(zip(self.param_names, params))
        return self.predict(t)
