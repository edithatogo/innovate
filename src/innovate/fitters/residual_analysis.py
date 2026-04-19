"""Residual analysis utilities for diffusion model diagnostics."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ResidualAnalysis:
    """Container for residual analysis results."""

    residuals: np.ndarray
    standardized_residuals: np.ndarray
    durbin_watson: float
    shapiro_wilk_p: float
    breusch_pagan_p: float | None
    mean_residual: float
    std_residual: float
    max_abs_residual: float
    residual_autocorrelation: np.ndarray

    def has_autocorrelation(self, threshold: float = 0.05) -> bool:
        """Check if residuals show significant autocorrelation (Durbin-Watson test)."""
        # Durbin-Watson statistic: values < 1.5 suggest positive autocorrelation
        return self.durbin_watson < 1.5 or self.durbin_watson > 2.5

    def is_normally_distributed(self, alpha: float = 0.05) -> bool:
        """Check if residuals are normally distributed (Shapiro-Wilk test)."""
        return self.shapiro_wilk_p > alpha

    def has_heteroscedasticity(self, alpha: float = 0.05) -> bool:
        """Check if residuals show heteroscedasticity (Breusch-Pagan test)."""
        if self.breusch_pagan_p is None:
            return False
        return self.breusch_pagan_p < alpha

    def summary(self) -> str:
        """Return a formatted summary of residual analysis."""
        lines = [
            "Residual Analysis Summary",
            "=" * 40,
            f"Mean:              {self.mean_residual:.6f}",
            f"Std Dev:           {self.std_residual:.6f}",
            f"Max |Residual|:    {self.max_abs_residual:.6f}",
            f"Durbin-Watson:     {self.durbin_watson:.4f}",
            f"Shapiro-Wilk p:    {self.shapiro_wilk_p:.6f}",
            f"Normality (alpha=0.05): {'Yes' if self.is_normally_distributed() else 'No'}",
            f"Autocorrelation:   {'Yes' if self.has_autocorrelation() else 'No'}",
        ]
        if self.breusch_pagan_p is not None:
            lines.append(f"Breusch-Pagan p:   {self.breusch_pagan_p:.6f}")
            lines.append(f"Heteroscedasticity: {'Yes' if self.has_heteroscedasticity() else 'No'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Serialize the residual analysis into JSON-friendly values."""
        return {
            "residuals": self.residuals.tolist(),
            "standardized_residuals": self.standardized_residuals.tolist(),
            "durbin_watson": self.durbin_watson,
            "shapiro_wilk_p": self.shapiro_wilk_p,
            "breusch_pagan_p": self.breusch_pagan_p,
            "mean_residual": self.mean_residual,
            "std_residual": self.std_residual,
            "max_abs_residual": self.max_abs_residual,
            "residual_autocorrelation": self.residual_autocorrelation.tolist(),
        }


def analyze_residuals(  # noqa: PLR0912
    residuals: np.ndarray,
    fitted_values: np.ndarray | None = None,
) -> ResidualAnalysis:
    """Perform comprehensive residual analysis.

    Args:
        residuals: Array of residuals (observed - predicted).
        fitted_values: Array of fitted/predicted values. If None, heteroscedasticity
                      test is skipped.

    Returns
    -------
        ResidualAnalysis object with diagnostic statistics.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)

    # Basic statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals, ddof=1) if n > 1 else 0.0
    max_abs_res = np.max(np.abs(residuals))

    # Standardized residuals
    if std_res > 0:
        std_residuals = (residuals - mean_res) / std_res
    else:
        std_residuals = np.zeros_like(residuals)

    # Durbin-Watson statistic for autocorrelation
    if n > 1:
        dw = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2))
    else:
        dw = 2.0  # No autocorrelation possible with single observation

    # Shapiro-Wilk test for normality
    if n >= 3 and n <= 5000:
        _, sw_p = stats.shapiro(residuals)
    elif n > 5000:
        # For large samples, use skewness/kurtosis test
        sw_p = float(stats.jarque_bera(residuals).pvalue)
    else:
        sw_p = float("nan")

    # Breusch-Pagan test for heteroscedasticity
    bp_p: float | None = None
    if fitted_values is not None and n > 4:
        fitted_values = np.asarray(fitted_values, dtype=float)
        if len(fitted_values) == n and np.std(fitted_values) > 0:
            try:
                # Simple implementation: regress squared residuals on fitted values
                squared_res = residuals**2
                slope, _, _, p_value, _ = stats.linregress(fitted_values, squared_res)
                bp_p = float(p_value)
            except Exception:
                bp_p = None

    # Lag-1 autocorrelation
    if n > 1:
        lag1_autocorr = np.correlate(residuals - mean_res, residuals - mean_res, mode="full")
        if lag1_autocorr[n - 1] > 0:
            lag1_autocorr = lag1_autocorr[n - 1 : n + 2] / lag1_autocorr[n - 1]
        else:
            lag1_autocorr = np.array([0.0, 1.0, 0.0])
    else:
        lag1_autocorr = np.array([0.0, 1.0, 0.0])

    return ResidualAnalysis(
        residuals=residuals,
        standardized_residuals=std_residuals,
        durbin_watson=dw,
        shapiro_wilk_p=sw_p,
        breusch_pagan_p=bp_p,
        mean_residual=float(mean_res),
        std_residual=float(std_res),
        max_abs_residual=float(max_abs_res),
        residual_autocorrelation=lag1_autocorr,
    )
