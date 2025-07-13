import pandas as pd
from causalimpact import CausalImpact

class CausalImpactAnalysis:
    """
    A wrapper for the CausalImpact library.
    """
    def __init__(self, data: pd.DataFrame, pre_period: list, post_period: list):
        self.data = data
        self.pre_period = pre_period
        self.post_period = post_period
        self.impact = None

    def run(self):
        """
        Runs the CausalImpact analysis.
        """
        self.impact = CausalImpact(self.data, self.pre_period, self.post_period)

    @property
    def summary(self):
        """
        Returns a summary of the CausalImpact analysis.
        """
        if self.impact is None:
            raise RuntimeError("CausalImpact analysis has not been run yet. Call .run() first.")
        return self.impact.summary()

    def plot(self):
        """
        Plots the results of the CausalImpact analysis.
        """
        if self.impact is None:
            raise RuntimeError("CausalImpact analysis has not been run yet. Call .run() first.")
        self.impact.plot()
