"""Scenario execution and comparison APIs.

This module provides APIs for executing scenarios, tracking execution
metadata and diagnostics, and comparing scenario results.
"""

import concurrent.futures
from datetime import UTC, datetime
from functools import partial
from typing import Any

import numpy as np

from innovate.scenario.schemas import ScenarioBase


class ScenarioExecution:
    """Container for a single scenario execution with results and metadata."""

    def __init__(
        self,
        scenario: ScenarioBase,
        seed: int,
        model_type: str,
        version: str,
        execution_time_seconds: float = 0.0,
        time_points: np.ndarray | None = None,
        adoption_curve: np.ndarray | None = None,
        diagnostics: dict[str, Any] | None = None,
        notes: str | None = None,
        timestamp: datetime | None = None,
    ):
        """Initialize a scenario execution.

        Parameters
        ----------
        scenario
            The scenario specification that was executed.
        seed
            Random seed for reproducibility.
        model_type
            Type of model used (e.g., 'bass', 'fisher_pry').
        version
            Version string of the implementation.
        execution_time_seconds
            Time taken to execute the scenario in seconds.
        time_points
            Time points at which adoption was evaluated (optional).
        adoption_curve
            Adoption values at each time point (optional).
        diagnostics
            Dictionary of diagnostic metrics (optional).
        notes
            Optional notes about the execution.
        timestamp
            Timestamp of execution (defaults to current time).
        """
        self.scenario = scenario
        self.seed = int(seed)
        self.model_type = str(model_type)
        self.version = str(version)
        self.execution_time_seconds = float(execution_time_seconds)
        self.time_points = np.array(time_points, dtype=float) if time_points is not None else None
        self.adoption_curve = np.array(adoption_curve, dtype=float) if adoption_curve is not None else None
        self.diagnostics = diagnostics if diagnostics is not None else {}
        self.notes = notes if notes is None else str(notes)
        self.timestamp = timestamp if timestamp is not None else datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert execution to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with all execution data.
        """
        return {
            "scenario": self.scenario.to_dict(),
            "seed": self.seed,
            "model_type": self.model_type,
            "version": self.version,
            "execution_time_seconds": self.execution_time_seconds,
            "time_points": (self.time_points.tolist() if self.time_points is not None else None),
            "adoption_curve": (self.adoption_curve.tolist() if self.adoption_curve is not None else None),
            "diagnostics": self.diagnostics,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }


class ScenarioComparison:
    """Container for comparing two scenario executions."""

    def __init__(
        self,
        baseline_execution: ScenarioExecution,
        alternative_execution: ScenarioExecution,
        comparison_metric: str = "ranking",
        incremental_effect: float | None = None,
        threshold_timing: dict[str, Any] | None = None,
        uncertainty_bounds: dict[str, float] | None = None,
        notes: str | None = None,
    ):
        """Initialize a scenario comparison.

        Parameters
        ----------
        baseline_execution
            Execution of the baseline scenario.
        alternative_execution
            Execution of the alternative scenario.
        comparison_metric
            Metric used for comparison (e.g., 'ranking', 'incremental_effect').
        incremental_effect
            Incremental effect size (optional).
        threshold_timing
            Timing of threshold crossings (optional).
        uncertainty_bounds
            Bounds on uncertainty (optional).
        notes
            Optional notes about the comparison.
        """
        self.baseline_execution = baseline_execution
        self.alternative_execution = alternative_execution
        self.comparison_metric = str(comparison_metric)
        self.incremental_effect = float(incremental_effect) if incremental_effect is not None else None
        self.threshold_timing = threshold_timing if threshold_timing is not None else {}
        self.uncertainty_bounds = uncertainty_bounds if uncertainty_bounds is not None else {}
        self.notes = notes if notes is None else str(notes)

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the comparison.
        """
        return {
            "baseline_execution": self.baseline_execution.to_dict(),
            "alternative_execution": self.alternative_execution.to_dict(),
            "comparison_metric": self.comparison_metric,
            "incremental_effect": self.incremental_effect,
            "threshold_timing": self.threshold_timing,
            "uncertainty_bounds": self.uncertainty_bounds,
            "notes": self.notes,
        }


class ScenarioExecutor:
    """Executor for running scenarios with reproducible seeding."""

    def __init__(
        self,
        model_type: str,
        version: str,
        seed: int = 42,
    ):
        """Initialize a scenario executor.

        Parameters
        ----------
        model_type
            Type of model to use for execution.
        version
            Version of the implementation.
        seed
            Random seed for reproducibility.
        """
        self.model_type = str(model_type)
        self.version = str(version)
        self.seed = int(seed)

    def execute(
        self,
        scenario: ScenarioBase,
        notes: str | None = None,
    ) -> ScenarioExecution:
        """Execute a single scenario.

        Parameters
        ----------
        scenario
            The scenario to execute.
        notes
            Optional notes about the execution.

        Returns
        -------
        ScenarioExecution
            The execution result with metadata.
        """
        return ScenarioExecution(
            scenario=scenario,
            seed=self.seed,
            model_type=self.model_type,
            version=self.version,
            notes=notes,
        )

    def execute_grid(
        self,
        scenarios: list[ScenarioBase],
        notes: str | None = None,
        max_workers: int | None = None,
    ) -> list[ScenarioExecution]:
        """Execute a grid of scenarios.

        Parameters
        ----------
        scenarios
            List of scenarios to execute.
        notes
            Optional notes about the grid execution.
        max_workers
            Optional max workers for ThreadPoolExecutor.

        Returns
        -------
        list[ScenarioExecution]
            List of execution results.
        """
        execute_partial = partial(self.execute, notes=notes)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(execute_partial, scenarios))


def compare_scenarios(
    baseline: ScenarioExecution,
    alternative: ScenarioExecution,
    metric: str = "ranking",
) -> ScenarioComparison:
    """Compare two scenario executions.

    Parameters
    ----------
    baseline
        Baseline scenario execution.
    alternative
        Alternative scenario execution.
    metric
        Comparison metric to use.

    Returns
    -------
    ScenarioComparison
        Comparison object.
    """
    return ScenarioComparison(
        baseline_execution=baseline,
        alternative_execution=alternative,
        comparison_metric=metric,
    )
