import numpy as np
import pandas as pd
from econml.dml import LinearDML

class EconMLAnalysis:
    """
    A wrapper for the EconML library.
    """
    def __init__(self, data: pd.DataFrame, outcome: str, treatment: str, instruments: list, covariates: list):
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
        self.instruments = instruments
        self.covariates = covariates
        self.model = None

    def fit(self):
        """
        Fits the EconML model.
        """
        Y = self.data[self.outcome]
        T = self.data[self.treatment]
        Z = self.data[self.instruments]
        X = self.data[self.covariates]

        self.model = LinearDML(model_y='linear_regression', model_t='linear_regression')
        self.model.fit(Y, T, X=X, Z=Z)

    def effect(self, X=None):
        """
        Calculates the causal effect.
        """
        if self.model is None:
            raise RuntimeError("EconML model has not been fitted yet. Call .fit() first.")
        return self.model.effect(X)

    @property
    def summary(self):
        """
        Returns a summary of the fitted model.
        """
        if self.model is None:
            raise RuntimeError("EconML model has not been fitted yet. Call .fit() first.")
        return self.model.summary()
