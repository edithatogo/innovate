from typing import Sequence, Dict, Any
from ..base import DiffusionModel


class CounterfactualAnalysis:
    """
    A class for conducting counterfactual analysis on fitted diffusion models.
    """

    def __init__(self, model: DiffusionModel):
        if not model.params_:
            raise ValueError(
                "The model must be fitted before conducting counterfactual analysis."
            )
        self.model = model
        self.baseline_forecast = None
        self.counterfactual_forecasts = {}

    def run_baseline(
        self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None
    ):
        """
        Generate the baseline forecast using the original fitted model.
        """
        self.baseline_forecast = self.model.predict(t, covariates)

    def run_counterfactual(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: Dict[str, Any] = None,
        counterfactual_covariates: Dict[str, Sequence[float]] = None,
    ):
        """
        Generate a forecast for a given counterfactual scenario.
        """
        # Create a deep copy of the model to avoid modifying the original
        import copy

        counterfactual_model = copy.deepcopy(self.model)

        # Update parameters for the counterfactual scenario
        if counterfactual_params:
            for param, value in counterfactual_params.items():
                if param in counterfactual_model.params_:
                    counterfactual_model.params_[param] = value
                else:
                    raise ValueError(f"Parameter '{param}' not found in the model.")

        # Use counterfactual covariates if provided, otherwise use original
        covariates_to_use = (
            counterfactual_covariates
            if counterfactual_covariates is not None
            else self.model.covariates
        )

        # Generate the counterfactual forecast
        forecast = counterfactual_model.predict(t, covariates_to_use)
        self.counterfactual_forecasts[scenario_name] = forecast

    def compare_scenarios(self, scenario_name: str):
        """
        Compare a counterfactual scenario to the baseline forecast.
        """
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first."
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = self.baseline_forecast
        counterfactual = self.counterfactual_forecasts[scenario_name]

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }
