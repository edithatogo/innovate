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

        # The function to be fitted is the model's predict method.
        # We need a wrapper because curve_fit expects a function where the first
        # argument is the independent variable (t) and the following arguments
        # are the parameters to be fitted.
        def fit_function(t, *params):
            # Create a temporary dictionary of parameters for the predict method
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t)

        try:
            # Use the public predict method for fitting
            popt, _ = curve_fit(fit_function, t_arr, y_arr, p0=p0, bounds=bounds, **kwargs)
            # Store the optimized parameters back into the model
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self