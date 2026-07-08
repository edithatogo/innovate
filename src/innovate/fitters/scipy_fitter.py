import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
from scipy.optimize import curve_fit, differential_evolution, least_squares, minimize

from innovate.base.base import DiffusionModel
from innovate.compete.competition import MultiProductDiffusionModel
from innovate.fitters.diagnostics_contract import DiagnosticsWarning, UncertaintySummary
from innovate.fitters.residual_analysis import analyze_residuals


@dataclass
class FitDiagnostics:
    """Container for fit diagnostics and goodness-of-fit metrics."""

    r_squared: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    fitted_params: dict[str, float] = field(default_factory=dict)
    n_observations: int = 0
    n_parameters: int = 0
    optimization_method: str = ""
    convergence_status: str = ""
    message: str = ""
    residual_analysis: object | None = None
    uncertainty: UncertaintySummary = field(default_factory=UncertaintySummary.point_estimate)
    warnings: list[DiagnosticsWarning] = field(default_factory=list)
    support_level: str = "supported"
    provenance: str = "deterministic"

    def summary(self) -> str:
        """Return a formatted summary of fit diagnostics."""
        lines = [
            "Fit Diagnostics Summary",
            "=" * 40,
            f"R²:              {self.r_squared:.6f}",
            f"RMSE:            {self.rmse:.6f}",
            f"MAE:             {self.mae:.6f}",
            f"AIC:             {self.aic:.4f}",
            f"BIC:             {self.bic:.4f}",
            f"Observations:    {self.n_observations}",
            f"Parameters:      {self.n_parameters}",
            f"Method:          {self.optimization_method}",
            f"Convergence:     {self.convergence_status}",
            f"Support level:   {self.support_level}",
            f"Uncertainty:     {self.uncertainty.report_type}",
        ]
        if self.warnings:
            lines.append(f"Warnings:        {len(self.warnings)}")
        if self.message:
            lines.append(f"Message:         {self.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Serialize fit diagnostics into a structured mapping."""
        return {
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "mae": self.mae,
            "aic": self.aic,
            "bic": self.bic,
            "residuals": self.residuals.tolist(),
            "fitted_params": self.fitted_params,
            "n_observations": self.n_observations,
            "n_parameters": self.n_parameters,
            "optimization_method": self.optimization_method,
            "convergence_status": self.convergence_status,
            "message": self.message,
            "residual_analysis": None
            if self.residual_analysis is None
            else getattr(self.residual_analysis, "to_dict", lambda: self.residual_analysis)(),
            "uncertainty": self.uncertainty.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "support_level": self.support_level,
            "provenance": self.provenance,
        }


OptimizationMethod = Literal["curve_fit", "least_squares", "nelder_mead", "lbfgsb", "differential_evolution", "auto"]



@dataclass
class FitConfig:
    """Configuration object for optimization routines."""

    model: DiffusionModel
    t_arr: np.ndarray
    y_arr: np.ndarray
    p0: list[float]
    bounds: tuple
    sigma: np.ndarray | None = None

class ScipyFitter:
    """A fitter class that uses SciPy optimization methods for model estimation.

    Supports multiple optimization methods including curve_fit, least_squares,
    Nelder-Mead, L-BFGS-B, and differential evolution. Provides goodness-of-fit
    diagnostics including R², RMSE, AIC, BIC, and residual analysis.
    """

    def __init__(
        self,
        method: OptimizationMethod = "auto",
        maxiter: int = 1000,
        tol: float = 1e-8,
        store_diagnostics: bool = True,
    ):
        """Initialize the ScipyFitter.

        Args:
            method: Optimization method to use. Options are:
                - "curve_fit": SciPy's curve_fit (default, uses Levenberg-Marquardt or TRF)
                - "least_squares": scipy.optimize.least_squares with robust loss functions
                - "nelder_mead": Nelder-Mead simplex method (derivative-free)
                - "lbfgsb": L-BFGS-B (bounded quasi-Newton)
                - "differential_evolution": Global optimization (slower but robust)
                - "auto": Automatically select based on problem characteristics
            maxiter: Maximum number of iterations for the optimizer.
            tol: Tolerance for convergence.
            store_diagnostics: Whether to store fit diagnostics after fitting.
        """
        self.method = method
        self.maxiter = maxiter
        self.tol = tol
        self.store_diagnostics = store_diagnostics
        self.diagnostics: FitDiagnostics | None = None

    def _select_method(self, model, t: np.ndarray, y: np.ndarray) -> str:
        """Automatically select the best optimization method for the problem."""
        if len(t) < 20:
            return "differential_evolution"
        n_params = len(model.param_names)
        if n_params > len(t) / 3:
            return "lbfgsb"
        return "curve_fit"

    def _compute_diagnostics(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        method: str,
        convergence_status: str = "converged",
        message: str = "",
    ) -> FitDiagnostics:
        """Compute goodness-of-fit diagnostics after successful fitting."""
        y_pred = model.predict(t)
        y_pred = np.asarray(y_pred).flatten()
        y_flat = y.flatten()

        residuals = y_flat - y_pred
        n = len(y_flat)
        k = len(model.param_names)

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(ss_res / n) if n > 0 else 0.0
        mae = np.mean(np.abs(residuals)) if n > 0 else 0.0
        residual_analysis = analyze_residuals(residuals, fitted_values=y_pred) if n > 1 else None

        warnings_list: list[DiagnosticsWarning] = []
        if convergence_status != "converged":
            warnings_list.append(
                DiagnosticsWarning(
                    code="optimizer_convergence",
                    message=message or "Optimization did not converge cleanly.",
                ),
            )

        # AIC and BIC (assuming Gaussian errors)
        if ss_res > 0 and n > k:
            log_likelihood = -n / 2 * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
        else:
            aic = np.inf
            bic = np.inf

        return FitDiagnostics(
            r_squared=float(r_squared),
            rmse=float(rmse),
            mae=float(mae),
            aic=float(aic),
            bic=float(bic),
            residuals=residuals,
            fitted_params=model.params_.copy(),
            n_observations=n,
            n_parameters=k,
            optimization_method=method,
            convergence_status=convergence_status,
            message=message,
            residual_analysis=residual_analysis,
            uncertainty=UncertaintySummary.point_estimate(),
            warnings=warnings_list,
            support_level="supported" if convergence_status == "converged" else "partial",
            provenance="deterministic",
        )

    def _fit_curve_fit(
        self,
        config: FitConfig,
        **kwargs,
    ) -> tuple[np.ndarray, str, str]:
        """Fit using scipy.optimize.curve_fit."""

        def fit_function(t, *params):
            config.model.params_ = dict(zip(config.model.param_names, params))
            return config.model.predict(t).flatten()

        popt, pcov = curve_fit(
            fit_function,
            config.t_arr,
            config.y_arr,
            p0=config.p0,
            bounds=config.bounds,
            sigma=config.sigma,
            absolute_sigma=True,
            maxfev=self.maxiter * 10,
            **kwargs,
        )
        return popt, "converged", "Optimization terminated successfully"

    def _fit_least_squares(
        self,
        config: FitConfig,
        **kwargs,
    ) -> tuple[np.ndarray, str, str]:
        """Fit using scipy.optimize.least_squares with robust loss."""

        def residuals(params):
            config.model.params_ = dict(zip(config.model.param_names, params))
            return config.model.predict(config.t_arr).flatten() - config.y_arr

        result = least_squares(
            residuals,
            config.p0,
            bounds=config.bounds,
            loss="huber",
            max_nfev=self.maxiter,
            **kwargs,
        )
        return result.x, "converged" if result.success else "failed", result.message

    def _fit_nelder_mead(
        self,
        config: FitConfig,
        **kwargs,
    ) -> tuple[np.ndarray, str, str]:
        """Fit using Nelder-Mead simplex method."""

        def objective(params):
            config.model.params_ = dict(zip(config.model.param_names, params))
            try:
                y_pred = config.model.predict(config.t_arr).flatten()
                return np.sum((config.y_arr - y_pred) ** 2)
            except Exception as e:
                warnings.warn(f"Objective evaluation failed during Nelder-Mead optimization: {e}", stacklevel=2)
                return 1e10

        result = minimize(
            objective,
            config.p0,
            method="Nelder-Mead",
            options={"maxiter": self.maxiter, "adaptive": True},
            **kwargs,
        )
        return result.x, "converged" if result.success else "failed", result.message

    def _fit_lbfgsb(
        self,
        config: FitConfig,
        **kwargs,
    ) -> tuple[np.ndarray, str, str]:
        """Fit using L-BFGS-B optimizer."""

        def objective(params):
            config.model.params_ = dict(zip(config.model.param_names, params))
            try:
                y_pred = config.model.predict(config.t_arr).flatten()
                return np.sum((config.y_arr - y_pred) ** 2)
            except Exception as e:
                warnings.warn(f"Objective evaluation failed during L-BFGS-B optimization: {e}", stacklevel=2)
                return 1e10

        lb, ub = config.bounds
        bounds_list = list(zip(lb, ub))

        result = minimize(
            objective,
            config.p0,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": self.maxiter},
            **kwargs,
        )
        return result.x, "converged" if result.success else "failed", result.message

    def _fit_differential_evolution(
        self,
        config: FitConfig,
        **kwargs,
    ) -> tuple[np.ndarray, str, str]:
        """Fit using differential evolution (global optimization)."""

        def objective(params):
            config.model.params_ = dict(zip(config.model.param_names, params))
            try:
                y_pred = config.model.predict(config.t_arr).flatten()
                return np.sum((config.y_arr - y_pred) ** 2)
            except Exception as e:
                warnings.warn(f"Objective evaluation failed during differential evolution: {e}", stacklevel=2)
                return 1e10

        lb, ub = config.bounds
        # Differential evolution requires finite config.bounds
        LARGE_BOUND = 1e6
        bounds_list = [(max(-LARGE_BOUND, lo), min(LARGE_BOUND, hi)) for lo, hi in zip(lb, ub)]

        result = differential_evolution(
            objective,
            bounds_list,
            maxiter=self.maxiter,
            tol=self.tol,
            polish=True,
            **kwargs,
        )
        return result.x, "converged" if result.success else "failed", result.message

    def fit(  # noqa: PLR0912
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using the configured optimization method.

        Args:
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments passed to the optimizer.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
            ValueError: If inputs are invalid.
        """
        # Input validation
        t_arr = np.asarray(t, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if len(t_arr) == 0 or len(y_arr) == 0:
            raise ValueError("Time and observation arrays must not be empty")
        if len(t_arr) != len(y_arr):
            raise ValueError(f"Time and observation arrays must have same length, got {len(t_arr)} and {len(y_arr)}")
        if np.any(~np.isfinite(y_arr)):
            raise ValueError("Observation array contains non-finite values (NaN or Inf)")
        if np.any(~np.isfinite(t_arr)):
            raise ValueError("Time array contains non-finite values (NaN or Inf)")

        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel
        if isinstance(model, MultiProductDiffusionModel):
            if weights is not None:
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            if bounds is not None:
                kwargs["bounds"] = bounds
            model.fit(t, y, **kwargs)
            if self.store_diagnostics:
                self.diagnostics = self._compute_diagnostics(model, t_arr, y_arr, "model_builtin")
            return self

        y_arr = y_arr.flatten()

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())
        else:
            p0 = list(p0)

        # Determine bounds if not provided
        if bounds is None:
            model_bounds = model.bounds(t, y)
            lower_bounds = [b[0] for b in model_bounds.values()]
            upper_bounds = [b[1] for b in model_bounds.values()]
            bounds = (lower_bounds, upper_bounds)
        else:
            lower_bounds, upper_bounds = bounds

        # Select optimization method
        method = self.method
        if method == "auto":
            method = self._select_method(model, t_arr, y_arr)

        fit_methods = {
            "curve_fit": self._fit_curve_fit,
            "least_squares": self._fit_least_squares,
            "nelder_mead": self._fit_nelder_mead,
            "lbfgsb": self._fit_lbfgsb,
            "differential_evolution": self._fit_differential_evolution,
        }

        if method not in fit_methods:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(fit_methods.keys())} or 'auto'")

        try:
            config = FitConfig(
                model=model,
                t_arr=t_arr,
                y_arr=y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
            )
            popt, status, message = fit_methods[method](config, **kwargs)
            model.params_ = dict(zip(model.param_names, popt))
        except Exception as e:
            raise RuntimeError(f"Fitting failed with method '{method}': {e}")

        # Compute and store diagnostics
        if self.store_diagnostics:
            self.diagnostics = self._compute_diagnostics(model, t_arr, y_arr, method, status, message)

        return self
