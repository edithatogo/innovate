from typing import Sequence, Dict, Callable, Any, Self
import numpy as np
from scipy.optimize import curve_fit
from ..models.base import DiffusionModel
from ..models.competition import MultiProductDiffusionModel

class ScipyFitter:
    """A fitter class that uses SciPy's curve_fit for model estimation."""

    def __init__(self):
        self.params_: Dict[str, float] = None

    def fit(self, model: DiffusionModel, t: Sequence[float], y: Sequence[float], **kwargs) -> Self:
        """
        Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns:
            A dictionary of the fitted parameters.
        
        Raises:
            RuntimeError: If fitting fails.
        """
        
        t_arr = np.array(t)
        y_arr = np.array(y)

        # For MultiProductDiffusionModel, y will be a flattened array of multiple products
        # We need to handle this by reshaping the prediction from the model.
        if isinstance(model, MultiProductDiffusionModel):
            # The objective function for curve_fit needs to take t and then the parameters
            # and return a flattened array of predictions for all products.
            
            # We need to define the parameters to be fitted for MultiProductDiffusionModel.
            # For now, let's assume we are fitting p, Q, m. This will require careful
            # flattening and unflattening of parameters.
            
            # This is a more complex scenario for curve_fit. A custom optimization routine
            # might be more suitable for MultiProductDiffusionModel.
            
            # For Phase 1, we will only support fitting of single-product models (Bass, Gompertz, Logistic)
            # with curve_fit directly. MultiProductDiffusionModel will be fitted by manually
            # setting parameters or with a more advanced fitter in later phases.
            
            raise NotImplementedError("Fitting MultiProductDiffusionModel with ScipyFitter is not yet implemented. "
                                      "Please use single-product models for now or manually set parameters for MultiProductDiffusionModel.")

        # For single-product models, we can use the model's predict method as the curve_fit function
        # The predict method expects (t, *params) if it's a static function, 
        # but here it's a method that uses internal state.
        # So, we need a wrapper function that takes parameters as arguments.
        
        # The model's _bass_cumulative, _gompertz_cumulative, _logistic_cumulative are suitable.
        # We need to get the initial guesses and bounds from the model itself or provide them here.
        
        # This requires a slight modification to how curve_fit is used with the DiffusionModel interface.
        # The DiffusionModel.fit() method already handles curve_fit internally for single models.
        # So, this ScipyFitter will primarily be used for more complex fitting scenarios or 
        # to provide a unified interface for different optimization backends.
        
        # For Phase 1, the single-product models (Bass, Gompertz, Logistic) already have their
        # own `fit` methods that use `curve_fit`. This `ScipyFitter` class will be more relevant
        # when we introduce a generic `fit` method that can take any `DiffusionModel` and optimize it.
        
        # Let's make this fitter work by calling the model's internal curve_fit logic
        # or by providing a generic way to get the curve function and initial parameters.
        
        # For now, this fitter will simply call the model's `fit` method.
        # This makes the `ScipyFitter` a thin wrapper for Phase 1.
        
        # In future phases, this `ScipyFitter` will contain the actual optimization logic
        # for various models, including MultiProductDiffusionModel.
        
        # Ensure the model has a fit method that accepts t and y
        if not hasattr(model, 'fit') or not callable(model.fit):
            raise AttributeError("Model must have a 'fit' method.")
        
        # Call the model's own fit method
        fitted_model = model.fit(t, y)
        self.params_ = fitted_model.params_
        return self
