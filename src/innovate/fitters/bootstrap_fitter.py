from typing import Sequence, Dict, List, Any
import numpy as np
from ..models.base import DiffusionModel

class BootstrapFitter:
    """A fitter class that uses bootstrapping to estimate parameter uncertainty."""

    def __init__(self, n_bootstraps: int = 100):
        self.n_bootstraps = n_bootstraps
        self.bootstrapped_params: List[Dict[str, float]] = []

    def fit(self, model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()
            
            try:
                boot_model.fit(t_resampled, y_resampled)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def get_parameter_estimates(self) -> Dict[str, List[float]]:
        """Returns a dictionary of parameter names to lists of bootstrapped values."""
        if not self.bootstrapped_params:
            return {}

        # Assuming all models have the same parameter names
        param_names = self.bootstrapped_params[0].keys()
        estimates = {name: [] for name in param_names}

        for params_dict in self.bootstrapped_params:
            for name, value in params_dict.items():
                estimates[name].append(value)
        return estimates

    def get_confidence_intervals(self, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def get_standard_errors(self) -> Dict[str, float]:
        """Returns standard errors for each parameter."""
        estimates = self.get_parameter_estimates()
        ses = {}
        for name, values in estimates.items():
            if values:
                ses[name] = float(np.std(values))
        return ses
