# src/innovate/fitters/curve_fitter.py

import numpy as np
from scipy.optimize import curve_fit
from innovate.base.base import DiffusionModel


class CurveFitter:
    """
    A fitter that uses scipy.optimize.curve_fit to estimate model parameters.
    """

    def __init__(self, model: DiffusionModel):
        self.model = model

    def fit(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """
        Fits the model to the data using curve_fit.
        """

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self
