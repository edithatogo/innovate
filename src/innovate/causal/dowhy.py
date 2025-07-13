import pandas as pd
import dowhy
from dowhy import CausalModel

class DoWhyAnalysis:
    """
    A wrapper for the DoWhy library.
    """
    def __init__(self, data: pd.DataFrame, treatment: str, outcome: str, graph: str):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.graph = graph
        self.model = None
        self.identified_estimand = None
        self.estimate = None

    def run(self):
        """
        Runs the DoWhy analysis.
        """
        self.model = CausalModel(
            data=self.data,
            treatment=self.treatment,
            outcome=self.outcome,
            graph=self.graph
        )

        self.identified_estimand = self.model.identify_effect()
        self.estimate = self.model.estimate_effect(self.identified_estimand, method_name="backdoor.linear_regression")

    @property
    def summary(self):
        """
        Returns a summary of the DoWhy analysis.
        """
        if self.estimate is None:
            raise RuntimeError("DoWhy analysis has not been run yet. Call .run() first.")
        return self.estimate
