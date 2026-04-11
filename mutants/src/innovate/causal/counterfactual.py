from collections.abc import Sequence
from typing import Any

import numpy as np

from ..base import DiffusionModel
from typing import Annotated
from typing import Callable
from typing import ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"] # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None): # type: ignore
    """Forward call to original or mutated function, depending on the environment"""
    import os # type: ignore
    mutant_under_test = os.environ['MUTANT_UNDER_TEST'] # type: ignore
    if mutant_under_test == 'fail': # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException # type: ignore
        raise MutmutProgrammaticFailException('Failed programmatically')       # type: ignore
    elif mutant_under_test == 'stats': # type: ignore
        from mutmut.__main__ import record_trampoline_hit # type: ignore
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__) # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_' # type: ignore
    if not mutant_under_test.startswith(prefix): # type: ignore
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    mutant_name = mutant_under_test.rpartition('.')[-1] # type: ignore
    if self_arg is not None: # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs) # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs) # type: ignore
    return result # type: ignore


class CounterfactualAnalysis:
    """A class for conducting counterfactual analysis on fitted diffusion models."""

    def __init__(self, model: "DiffusionModel"):
        args = [model]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCounterfactualAnalysisǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁCounterfactualAnalysisǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁCounterfactualAnalysisǁ__init____mutmut_orig(self, model: "DiffusionModel"):
        if not model.params_:
            raise ValueError(
                "The model must be fitted before conducting counterfactual analysis.",
            )
        self.model = model
        self.baseline_forecast: Sequence[float] | None = None
        self.counterfactual_forecasts: dict[str, Sequence[float]] = {}

    def xǁCounterfactualAnalysisǁ__init____mutmut_1(self, model: "DiffusionModel"):
        if model.params_:
            raise ValueError(
                "The model must be fitted before conducting counterfactual analysis.",
            )
        self.model = model
        self.baseline_forecast: Sequence[float] | None = None
        self.counterfactual_forecasts: dict[str, Sequence[float]] = {}

    def xǁCounterfactualAnalysisǁ__init____mutmut_2(self, model: "DiffusionModel"):
        if not model.params_:
            raise ValueError(
                None,
            )
        self.model = model
        self.baseline_forecast: Sequence[float] | None = None
        self.counterfactual_forecasts: dict[str, Sequence[float]] = {}

    def xǁCounterfactualAnalysisǁ__init____mutmut_3(self, model: "DiffusionModel"):
        if not model.params_:
            raise ValueError(
                "XXThe model must be fitted before conducting counterfactual analysis.XX",
            )
        self.model = model
        self.baseline_forecast: Sequence[float] | None = None
        self.counterfactual_forecasts: dict[str, Sequence[float]] = {}

    def xǁCounterfactualAnalysisǁ__init____mutmut_4(self, model: "DiffusionModel"):
        if not model.params_:
            raise ValueError(
                "the model must be fitted before conducting counterfactual analysis.",
            )
        self.model = model
        self.baseline_forecast: Sequence[float] | None = None
        self.counterfactual_forecasts: dict[str, Sequence[float]] = {}

    def xǁCounterfactualAnalysisǁ__init____mutmut_5(self, model: "DiffusionModel"):
        if not model.params_:
            raise ValueError(
                "THE MODEL MUST BE FITTED BEFORE CONDUCTING COUNTERFACTUAL ANALYSIS.",
            )
        self.model = model
        self.baseline_forecast: Sequence[float] | None = None
        self.counterfactual_forecasts: dict[str, Sequence[float]] = {}

    def xǁCounterfactualAnalysisǁ__init____mutmut_6(self, model: "DiffusionModel"):
        if not model.params_:
            raise ValueError(
                "The model must be fitted before conducting counterfactual analysis.",
            )
        self.model = None
        self.baseline_forecast: Sequence[float] | None = None
        self.counterfactual_forecasts: dict[str, Sequence[float]] = {}

    def xǁCounterfactualAnalysisǁ__init____mutmut_7(self, model: "DiffusionModel"):
        if not model.params_:
            raise ValueError(
                "The model must be fitted before conducting counterfactual analysis.",
            )
        self.model = model
        self.baseline_forecast: Sequence[float] | None = ""
        self.counterfactual_forecasts: dict[str, Sequence[float]] = {}

    def xǁCounterfactualAnalysisǁ__init____mutmut_8(self, model: "DiffusionModel"):
        if not model.params_:
            raise ValueError(
                "The model must be fitted before conducting counterfactual analysis.",
            )
        self.model = model
        self.baseline_forecast: Sequence[float] | None = None
        self.counterfactual_forecasts: dict[str, Sequence[float]] = None
    
    xǁCounterfactualAnalysisǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCounterfactualAnalysisǁ__init____mutmut_1': xǁCounterfactualAnalysisǁ__init____mutmut_1, 
        'xǁCounterfactualAnalysisǁ__init____mutmut_2': xǁCounterfactualAnalysisǁ__init____mutmut_2, 
        'xǁCounterfactualAnalysisǁ__init____mutmut_3': xǁCounterfactualAnalysisǁ__init____mutmut_3, 
        'xǁCounterfactualAnalysisǁ__init____mutmut_4': xǁCounterfactualAnalysisǁ__init____mutmut_4, 
        'xǁCounterfactualAnalysisǁ__init____mutmut_5': xǁCounterfactualAnalysisǁ__init____mutmut_5, 
        'xǁCounterfactualAnalysisǁ__init____mutmut_6': xǁCounterfactualAnalysisǁ__init____mutmut_6, 
        'xǁCounterfactualAnalysisǁ__init____mutmut_7': xǁCounterfactualAnalysisǁ__init____mutmut_7, 
        'xǁCounterfactualAnalysisǁ__init____mutmut_8': xǁCounterfactualAnalysisǁ__init____mutmut_8
    }
    xǁCounterfactualAnalysisǁ__init____mutmut_orig.__name__ = 'xǁCounterfactualAnalysisǁ__init__'

    def run_baseline(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCounterfactualAnalysisǁrun_baseline__mutmut_orig'), object.__getattribute__(self, 'xǁCounterfactualAnalysisǁrun_baseline__mutmut_mutants'), args, kwargs, self)

    def xǁCounterfactualAnalysisǁrun_baseline__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate the baseline forecast using the original fitted model."""
        self.baseline_forecast = self.model.predict(t)

    def xǁCounterfactualAnalysisǁrun_baseline__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate the baseline forecast using the original fitted model."""
        self.baseline_forecast = None

    def xǁCounterfactualAnalysisǁrun_baseline__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate the baseline forecast using the original fitted model."""
        self.baseline_forecast = self.model.predict(None)
    
    xǁCounterfactualAnalysisǁrun_baseline__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCounterfactualAnalysisǁrun_baseline__mutmut_1': xǁCounterfactualAnalysisǁrun_baseline__mutmut_1, 
        'xǁCounterfactualAnalysisǁrun_baseline__mutmut_2': xǁCounterfactualAnalysisǁrun_baseline__mutmut_2
    }
    xǁCounterfactualAnalysisǁrun_baseline__mutmut_orig.__name__ = 'xǁCounterfactualAnalysisǁrun_baseline'

    def run_counterfactual(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        args = [scenario_name, t, counterfactual_params, counterfactual_covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_orig'), object.__getattribute__(self, 'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_mutants'), args, kwargs, self)

    def xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_orig(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate a forecast for a given counterfactual scenario."""
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

        # Generate the counterfactual forecast
        forecast = counterfactual_model.predict(t)
        self.counterfactual_forecasts[scenario_name] = forecast

    def xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_1(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate a forecast for a given counterfactual scenario."""
        # Create a deep copy of the model to avoid modifying the original
        import copy

        counterfactual_model = None

        # Update parameters for the counterfactual scenario
        if counterfactual_params:
            for param, value in counterfactual_params.items():
                if param in counterfactual_model.params_:
                    counterfactual_model.params_[param] = value
                else:
                    raise ValueError(f"Parameter '{param}' not found in the model.")

        # Generate the counterfactual forecast
        forecast = counterfactual_model.predict(t)
        self.counterfactual_forecasts[scenario_name] = forecast

    def xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_2(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate a forecast for a given counterfactual scenario."""
        # Create a deep copy of the model to avoid modifying the original
        import copy

        counterfactual_model = copy.deepcopy(None)

        # Update parameters for the counterfactual scenario
        if counterfactual_params:
            for param, value in counterfactual_params.items():
                if param in counterfactual_model.params_:
                    counterfactual_model.params_[param] = value
                else:
                    raise ValueError(f"Parameter '{param}' not found in the model.")

        # Generate the counterfactual forecast
        forecast = counterfactual_model.predict(t)
        self.counterfactual_forecasts[scenario_name] = forecast

    def xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_3(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate a forecast for a given counterfactual scenario."""
        # Create a deep copy of the model to avoid modifying the original
        import copy

        counterfactual_model = copy.copy(self.model)

        # Update parameters for the counterfactual scenario
        if counterfactual_params:
            for param, value in counterfactual_params.items():
                if param in counterfactual_model.params_:
                    counterfactual_model.params_[param] = value
                else:
                    raise ValueError(f"Parameter '{param}' not found in the model.")

        # Generate the counterfactual forecast
        forecast = counterfactual_model.predict(t)
        self.counterfactual_forecasts[scenario_name] = forecast

    def xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_4(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate a forecast for a given counterfactual scenario."""
        # Create a deep copy of the model to avoid modifying the original
        import copy

        counterfactual_model = copy.deepcopy(self.model)

        # Update parameters for the counterfactual scenario
        if counterfactual_params:
            for param, value in counterfactual_params.items():
                if param not in counterfactual_model.params_:
                    counterfactual_model.params_[param] = value
                else:
                    raise ValueError(f"Parameter '{param}' not found in the model.")

        # Generate the counterfactual forecast
        forecast = counterfactual_model.predict(t)
        self.counterfactual_forecasts[scenario_name] = forecast

    def xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_5(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate a forecast for a given counterfactual scenario."""
        # Create a deep copy of the model to avoid modifying the original
        import copy

        counterfactual_model = copy.deepcopy(self.model)

        # Update parameters for the counterfactual scenario
        if counterfactual_params:
            for param, value in counterfactual_params.items():
                if param in counterfactual_model.params_:
                    counterfactual_model.params_[param] = None
                else:
                    raise ValueError(f"Parameter '{param}' not found in the model.")

        # Generate the counterfactual forecast
        forecast = counterfactual_model.predict(t)
        self.counterfactual_forecasts[scenario_name] = forecast

    def xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_6(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate a forecast for a given counterfactual scenario."""
        # Create a deep copy of the model to avoid modifying the original
        import copy

        counterfactual_model = copy.deepcopy(self.model)

        # Update parameters for the counterfactual scenario
        if counterfactual_params:
            for param, value in counterfactual_params.items():
                if param in counterfactual_model.params_:
                    counterfactual_model.params_[param] = value
                else:
                    raise ValueError(None)

        # Generate the counterfactual forecast
        forecast = counterfactual_model.predict(t)
        self.counterfactual_forecasts[scenario_name] = forecast

    def xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_7(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate a forecast for a given counterfactual scenario."""
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

        # Generate the counterfactual forecast
        forecast = None
        self.counterfactual_forecasts[scenario_name] = forecast

    def xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_8(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate a forecast for a given counterfactual scenario."""
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

        # Generate the counterfactual forecast
        forecast = counterfactual_model.predict(None)
        self.counterfactual_forecasts[scenario_name] = forecast

    def xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_9(
        self,
        scenario_name: str,
        t: Sequence[float],
        counterfactual_params: dict[str, Any] | None = None,
        counterfactual_covariates: dict[str, Sequence[float]] | None = None,
    ):
        """Generate a forecast for a given counterfactual scenario."""
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

        # Generate the counterfactual forecast
        forecast = counterfactual_model.predict(t)
        self.counterfactual_forecasts[scenario_name] = None
    
    xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_1': xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_1, 
        'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_2': xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_2, 
        'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_3': xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_3, 
        'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_4': xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_4, 
        'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_5': xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_5, 
        'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_6': xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_6, 
        'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_7': xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_7, 
        'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_8': xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_8, 
        'xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_9': xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_9
    }
    xǁCounterfactualAnalysisǁrun_counterfactual__mutmut_orig.__name__ = 'xǁCounterfactualAnalysisǁrun_counterfactual'

    def compare_scenarios(self, scenario_name: str):
        args = [scenario_name]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_orig'), object.__getattribute__(self, 'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_mutants'), args, kwargs, self)

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_orig(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_1(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is not None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_2(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                None,
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_3(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "XXBaseline forecast has not been run. Call .run_baseline() first.XX",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_4(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "baseline forecast has not been run. call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_5(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "BASELINE FORECAST HAS NOT BEEN RUN. CALL .RUN_BASELINE() FIRST.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_6(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_7(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(None)

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_8(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = None
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_9(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(None)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_10(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = None

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_11(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(None)

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_12(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = None
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_13(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual + baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_14(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = None

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_15(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) / 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_16(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference * baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_17(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 101

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_18(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "XXbaselineXX": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_19(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "BASELINE": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_20(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "XXcounterfactualXX": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_21(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "COUNTERFACTUAL": counterfactual,
            "difference": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_22(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "XXdifferenceXX": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_23(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "DIFFERENCE": difference,
            "percentage_difference": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_24(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "XXpercentage_differenceXX": percentage_difference,
        }

    def xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_25(self, scenario_name: str):
        """Compare a counterfactual scenario to the baseline forecast."""
        if self.baseline_forecast is None:
            raise RuntimeError(
                "Baseline forecast has not been run. Call .run_baseline() first.",
            )
        if scenario_name not in self.counterfactual_forecasts:
            raise ValueError(f"Counterfactual scenario '{scenario_name}' not found.")

        baseline = np.array(self.baseline_forecast)
        counterfactual = np.array(self.counterfactual_forecasts[scenario_name])

        # Calculate the difference and percentage difference
        difference = counterfactual - baseline
        percentage_difference = (difference / baseline) * 100

        return {
            "baseline": baseline,
            "counterfactual": counterfactual,
            "difference": difference,
            "PERCENTAGE_DIFFERENCE": percentage_difference,
        }
    
    xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_1': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_1, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_2': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_2, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_3': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_3, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_4': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_4, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_5': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_5, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_6': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_6, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_7': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_7, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_8': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_8, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_9': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_9, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_10': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_10, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_11': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_11, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_12': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_12, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_13': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_13, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_14': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_14, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_15': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_15, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_16': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_16, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_17': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_17, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_18': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_18, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_19': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_19, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_20': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_20, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_21': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_21, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_22': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_22, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_23': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_23, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_24': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_24, 
        'xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_25': xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_25
    }
    xǁCounterfactualAnalysisǁcompare_scenarios__mutmut_orig.__name__ = 'xǁCounterfactualAnalysisǁcompare_scenarios'
