from typing import Sequence, Dict, Callable, Any, Self
import numpy as np
from scipy.optimize import curve_fit
from innovate.base.base import DiffusionModel
from innovate.compete.competition import MultiProductDiffusionModel # Import the model

class ScipyFitter:
    """A fitter class that uses SciPy's curve_fit for model estimation."""

    def fit(self, model: DiffusionModel, t: Sequence[float], y: Sequence[float], p0: Sequence[float] = None, bounds: tuple = None, covariates: Dict[str, Sequence[float]] = None, **kwargs) -> Self:
        """
        Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            covariates: A dictionary of covariate names and their values.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns:
            The fitter instance.
        
        Raises:
            RuntimeError: If fitting fails.
        """
        # Check for MultiProductDiffusionModel and raise NotImplementedError
        if isinstance(model, MultiProductDiffusionModel):
            raise NotImplementedError("Fitting MultiProductDiffusionModel with ScipyFitter is not yet implemented")
        
        t_arr = np.array(t)
        y_arr = np.array(y).flatten()

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())
            
        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t, covariates).flatten()

        try:
            popt, _ = curve_fit(fit_function, t_arr, y_arr, p0=p0, bounds=bounds, **kwargs)
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self