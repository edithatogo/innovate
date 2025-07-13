from ..base import DiffusionModel
from typing import Sequence, Dict
import pandas as pd

class SurvivalModel:
    """
    A wrapper for survival analysis models from the lifelines library.
    """
    def __init__(self, model_name: str = "Weibull"):
        try:
            import lifelines
        except ImportError:
            raise ImportError("The 'lifelines' library is required for this feature. Please install it with 'pip install lifelines'.")

        if model_name == "Weibull":
            self.model = lifelines.WeibullFitter()
        elif model_name == "Exponential":
            self.model = lifelines.ExponentialFitter()
        elif model_name == "LogNormal":
            self.model = lifelines.LogNormalFitter()
        elif model_name == "LogLogistic":
            self.model = lifelines.LogLogisticFitter()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def fit(self, durations: Sequence[float], event_observed: Sequence[int] = None, **kwargs):
        """
        Fits the survival model to the given durations and event data.
        """
        self.model.fit(durations, event_observed=event_observed, **kwargs)

    def predict(self, times: Sequence[float]):
        """
        Predicts the survival probability at the given times.
        """
        return self.model.predict(times)

    @property
    def summary(self):
        """
        Returns a summary of the fitted model.
        """
        return self.model.summary
