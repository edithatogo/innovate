import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from innovate.base.base import DiffusionModel
from innovate.base.options import FitOptions
from innovate.fitters.diagnostics_contract import (
    DiagnosticsContract,
    DiagnosticsWarning,
    UncertaintySummary,
    build_diagnostics_contract,
)


class BootstrapFitter:
    """A fitter class that uses bootstrapping to estimate parameter uncertainty."""

    def __init__(
        self,
        fitter: Any | None = None,
        n_bootstraps: int = 100,
        seed: int | None = None,
        n_bootstrap: int | None = None,
    ):
        """Initialize the BootstrapFitter.

        Args:
            fitter: The underlying fitter to use for each bootstrap sample (e.g., ScipyFitter).
            n_bootstraps: Number of bootstrap samples to generate.
            seed: Random seed for reproducibility.
        """
        if fitter is None:
            from innovate.fitters.scipy_fitter import ScipyFitter

            fitter = ScipyFitter()
        if n_bootstrap is not None:
            n_bootstraps = n_bootstrap

        self.fitter = fitter
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.bootstrapped_params: list[dict[str, float]] = []
        self.bootstrapped_diagnostics: list[Any] = []

    def fit(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],

        options: FitOptions | None = None,        **kwargs,
    ) -> None:
        """Perform bootstrap fitting to estimate parameter uncertainty.

        Args:
            model: The diffusion model to fit.
            t: Time points.
            y: Observed adoption data.
            **kwargs: Additional arguments passed to the underlying fitter.
        """
        rng = np.random.default_rng(self.seed)
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        self.bootstrapped_params = []
        self.bootstrapped_diagnostics = []

        for i in range(self.n_bootstraps):
            # Resample data with replacement
            indices = rng.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]
            order = np.argsort(t_resampled, kind="stable")
            t_resampled = t_resampled[order]
            y_resampled = y_resampled[order]

            # Create a new model instance for each bootstrap iteration
            boot_model = type(model)()

            try:
                self.fitter.fit(boot_model, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_.copy())
                # Store diagnostics if available
                if hasattr(self.fitter, "diagnostics") and self.fitter.diagnostics is not None:
                    self.bootstrapped_diagnostics.append(self.fitter.diagnostics)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                warnings.warn(f"Fitting failed for bootstrap sample {i}: {e}", stacklevel=2)
                continue

        if self.bootstrapped_params:
            estimates = self.get_parameter_estimates()
            model.params_ = {name: float(np.median(values)) for name, values in estimates.items() if values}

    def get_parameter_estimates(self) -> dict[str, list[float]]:
        """Returns a dictionary of parameter names to lists of bootstrapped values."""
        if not self.bootstrapped_params:
            return {}

        param_names = self.bootstrapped_params[0].keys()
        estimates: dict[str, list[float]] = {name: [] for name in param_names}

        for params_dict in self.bootstrapped_params:
            for name, value in params_dict.items():
                estimates[name].append(value)
        return estimates

    def get_confidence_intervals(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter.

        Args:
            alpha: Significance level (default 0.05 for 95% CI).

        Returns
        -------
            Dictionary mapping parameter names to dicts with 'lower', 'upper', and 'median' keys.
        """
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = float(np.percentile(values, (alpha / 2) * 100))
                upper = float(np.percentile(values, (1 - alpha / 2) * 100))
                median = float(np.percentile(values, 50))
                cis[name] = {"lower": lower, "upper": upper, "median": median}
        return cis

    def get_uncertainty_summary(self, alpha: float = 0.05) -> UncertaintySummary:
        """Return the bootstrap uncertainty summary in the shared contract format."""
        if not self.bootstrapped_params:
            return UncertaintySummary.unsupported("No bootstrap results available.", provenance="bootstrap")

        cis = self.get_confidence_intervals(alpha)
        lower = {name: values["lower"] for name, values in cis.items()}
        upper = {name: values["upper"] for name, values in cis.items()}
        median = {name: values["median"] for name, values in cis.items()}
        return UncertaintySummary.bootstrap_interval(
            lower=lower,
            upper=upper,
            median=median,
            level=1 - alpha,
            note=f"{len(self.bootstrapped_params)} successful bootstrap samples",
        )

    def get_standard_errors(self) -> dict[str, float]:
        """Returns standard errors for each parameter."""
        estimates = self.get_parameter_estimates()
        ses = {}
        for name, values in estimates.items():
            if values:
                ses[name] = float(np.std(values, ddof=1))
        return ses

    def get_parameter_correlation(self) -> dict[str, dict[str, float]] | None:
        """Returns the correlation matrix of parameter estimates."""
        estimates = self.get_parameter_estimates()
        if not estimates or len(estimates) < 2:
            return None

        param_names = list(estimates.keys())
        param_values = np.array([estimates[name] for name in param_names])

        if param_values.shape[1] < 2:
            return None

        corr_matrix = np.corrcoef(param_values)

        correlation: dict[str, dict[str, float]] = {}
        for i, name_i in enumerate(param_names):
            correlation[name_i] = {}
            for j, name_j in enumerate(param_names):
                correlation[name_i][name_j] = float(corr_matrix[i, j])

        return correlation

    def get_diagnostics_contract(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        *,
        alpha: float = 0.05,
        model_name: str = "",
    ) -> DiagnosticsContract:
        """Return a shared diagnostics contract for the bootstrapped model."""
        uncertainty = self.get_uncertainty_summary(alpha)
        warnings_list: list[DiagnosticsWarning] = []
        if uncertainty.support_level == "unsupported":
            warnings_list.append(
                DiagnosticsWarning(
                    code="bootstrap_unavailable",
                    message=uncertainty.note or "No bootstrap results were generated.",
                ),
            )

        return build_diagnostics_contract(
            model,
            t,
            y,
            provenance="bootstrap",
            uncertainty=uncertainty,
            warnings=warnings_list,
            model_name=model_name,
        )

    def summary(self, alpha: float = 0.05) -> str:
        """Return a formatted summary of bootstrap results."""
        if not self.bootstrapped_params:
            return "No bootstrap results available."

        cis = self.get_confidence_intervals(alpha)
        ses = self.get_standard_errors()
        n_successful = len(self.bootstrapped_params)

        lines = [
            "Bootstrap Parameter Estimates",
            "=" * 60,
            f"Successful bootstrap samples: {n_successful}/{self.n_bootstraps}",
            f"Confidence level: {(1 - alpha) * 100:.0f}%",
            "",
            f"{'Parameter':<20} {'Median':>12} {'Std Error':>12} {'95% CI Lower':>12} {'95% CI Upper':>12}",
            "-" * 60,
        ]

        for name in cis:
            median = cis[name]["median"]
            se = ses.get(name, 0.0)
            lower = cis[name]["lower"]
            upper = cis[name]["upper"]
            lines.append(f"{name:<20} {median:>12.6f} {se:>12.6f} {lower:>12.6f} {upper:>12.6f}")

        return "\n".join(lines)
