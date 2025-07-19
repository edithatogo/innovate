from typing import Sequence, Dict, Callable, Any, Self
import numpy as np
from scipy.optimize import curve_fit
from innovate.base.base import DiffusionModel
from innovate.compete.competition import MultiProductDiffusionModel # Import the model

class ScipyFitter:
    """A fitter class that uses SciPy's curve_fit for model estimation."""

    def fit(self, model: DiffusionModel, t: Sequence[float], y: Sequence[float], p0: Sequence[float] = None, bounds: tuple = None, **kwargs) -> Self:
        """
        Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns:
            The fitter instance.
        
        Raises:
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            y_arr = y_arr.flatten()
            # Create a dummy xdata of the same length as the flattened y_arr
            x_dummy = np.arange(len(y_arr))
            
            def fit_function(x_dummy_ignored, *params):
                param_dict = dict(zip(model.param_names, params))
                model.params_ = param_dict
                # The real t_arr is captured from the outer scope
                return model.predict(t_arr).flatten()
            
            # Use the dummy xdata for the curve_fit call
            x_fit = x_dummy
        else:
            y_arr = y_arr.flatten()
            def fit_function(t, *params):
                param_dict = dict(zip(model.param_names, params))
                model.params_ = param_dict
                return model.predict(t).flatten()
            x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())
            
        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(fit_function, x_fit, y_arr, p0=p0, bounds=bounds, **kwargs)
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self
