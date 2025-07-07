from typing import Sequence, Dict, Callable, Any, Self
import numpy as np
from scipy.optimize import curve_fit
from ..models.base import DiffusionModel

class ScipyFitter:
    """A fitter class that uses SciPy's curve_fit for model estimation."""

    def fit(self, model: DiffusionModel, t: Sequence[float], y: Sequence[float], p0: Sequence[float] = None, bounds: tuple = None, **kwargs) -> Self:
        """
        Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters.
            bounds: Bounds for the parameters.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns:
            The fitter instance.
        
        Raises:
            RuntimeError: If fitting fails.
        """
        
        t_arr = np.array(t)
        y_arr = np.array(y)

        if not hasattr(model, 'cumulative_adoption'):
            raise AttributeError("Model must have a 'cumulative_adoption' method.")

        try:
            params, _ = curve_fit(model.cumulative_adoption, t_arr, y_arr, 
                                  p0=p0, 
                                  bounds=bounds,
                                  maxfev=kwargs.get('maxfev', 5000),
                                  **kwargs)
            model.params_ = dict(zip(model.param_names, params))
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}. Try different initial guesses or check data.")

        return self