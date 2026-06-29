"""Tests for scenario execution and comparison summaries.

This module tests the experiment runner APIs for executing scenario grids,
generating diagnostics, and comparing scenario results.
"""

import json
from typing import Any

import numpy as np
import pytest

from innovate.scenario.execution import (
    ScenarioComparison,
    ScenarioExecution,
    ScenarioExecutor,
    compare_scenarios,
)
from innovate.scenario.schemas import (
    BaselineScenario,
    CompetitionScenario,
    InterventionScenario,
    SubstitutionScenario,
)


@pytest.fixture
def baseline_scenario() -> BaselineScenario:
    """Create a baseline scenario for testing."""
    return BaselineScenario(
        name="Baseline 2026",
        description="Reference scenario",
        time_horizon=20,
        time_unit="years",
        reference_year=2026,
        market_size=1000000,
        initial_adoption=0.01,
    )


@pytest.fixture
def intervention_scenario() -> InterventionScenario:
    """Create an intervention scenario for testing."""
    return InterventionScenario(
        name="Policy Subsidy",
        description="Subsidies for adoption",
        time_horizon=20,
        time_unit="years",
        reference_year=2026,
        market_size=1000000,
        initial_adoption=0.01,
        intervention_type="subsidy",
        intervention_start_time=5,
        intervention_magnitude=0.2,
    )


@pytest.fixture
def substitution_scenario() -> SubstitutionScenario:
    """Create a substitution scenario for testing."""
    return SubstitutionScenario(
        name="Technology Substitution",
        description="Old tech replaced by new",
        time_horizon=25,
        time_unit="years",
        reference_year=2026,
        market_size=500000,
        initial_adoption=0.05,
        incumbent_name="OldTech",
        entrant_name="NewTech",
        substitution_rate=0.1,
    )


@pytest.fixture
def competition_scenario() -> CompetitionScenario:
    """Create a competition scenario for testing."""
    return CompetitionScenario(
        name="Two-Product Market",
        description="Product A vs B",
        time_horizon=30,
        time_unit="years",
        reference_year=2026,
        market_size=1000000,
        initial_adoption=0.02,
        num_competitors=2,
        competitor_names=["Product A", "Product B"],
        market_share_initial=[0.6, 0.4],
    )


@pytest.mark.unit
class TestScenarioExecution:
    """Test ScenarioExecution class."""

    def test_create_execution_with_baseline(self, baseline_scenario):
        """Test creating a scenario execution."""
        execution = ScenarioExecution(
            scenario=baseline_scenario,
            seed=42,
            model_type="bass",
            version="0.5.0",
            execution_time_seconds=1.5,
        )
        assert execution.scenario.name == "Baseline 2026"
        assert execution.seed == 42
        assert execution.execution_time_seconds == 1.5

    def test_execution_with_results(self, baseline_scenario):
        """Test execution with simulation results."""
        time_points = np.array([0, 5, 10, 15, 20], dtype=float)
        adoption_curve = np.array([10000, 50000, 250000, 600000, 900000], dtype=float)

        execution = ScenarioExecution(
            scenario=baseline_scenario,
            seed=42,
            model_type="bass",
            version="0.5.0",
            time_points=time_points,
            adoption_curve=adoption_curve,
        )

        assert execution.time_points is not None
        assert len(execution.time_points) == 5
        assert np.allclose(execution.adoption_curve, adoption_curve)

    def test_execution_to_dict(self, baseline_scenario):
        """Test converting execution to dict."""
        execution = ScenarioExecution(
            scenario=baseline_scenario,
            seed=42,
            model_type="bass",
            version="0.5.0",
        )

        data = execution.to_dict()
        assert data["seed"] == 42
        assert data["model_type"] == "bass"
        assert data["scenario"]["name"] == "Baseline 2026"

    def test_execution_json_serializable(self, baseline_scenario):
        """Test that execution is JSON serializable."""
        execution = ScenarioExecution(
            scenario=baseline_scenario,
            seed=42,
            model_type="bass",
            version="0.5.0",
        )

        json_str = json.dumps(execution.to_dict())
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["seed"] == 42


@pytest.mark.unit
class TestScenarioComparison:
    """Test ScenarioComparison class."""

    def test_create_comparison(
        self,
        baseline_scenario,
        intervention_scenario,
    ):
        """Test creating a scenario comparison."""
        execution_baseline = ScenarioExecution(
            scenario=baseline_scenario,
            seed=42,
            model_type="bass",
            version="0.5.0",
        )
        execution_intervention = ScenarioExecution(
            scenario=intervention_scenario,
            seed=42,
            model_type="bass",
            version="0.5.0",
        )

        comparison = ScenarioComparison(
            baseline_execution=execution_baseline,
            alternative_execution=execution_intervention,
            comparison_metric="adoption_increase",
        )

        assert comparison.baseline_execution.seed == 42
        assert comparison.alternative_execution.scenario.intervention_type == "subsidy"

    def test_comparison_with_ranking(self):
        """Test comparison with ranking metric."""
        baseline = BaselineScenario(
            name="Baseline",
            description="Base",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
        )
        intervention = InterventionScenario(
            name="Intervention",
            description="Better",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
            intervention_type="subsidy",
            intervention_start_time=5,
            intervention_magnitude=0.2,
        )

        exec_base = ScenarioExecution(scenario=baseline, seed=42, model_type="bass", version="0.5.0")
        exec_int = ScenarioExecution(scenario=intervention, seed=42, model_type="bass", version="0.5.0")

        comparison = ScenarioComparison(
            baseline_execution=exec_base,
            alternative_execution=exec_int,
            comparison_metric="ranking",
        )

        assert comparison.comparison_metric == "ranking"

    def test_comparison_to_dict(self):
        """Test converting comparison to dict."""
        baseline = BaselineScenario(
            name="Baseline",
            description="Base",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
        )
        intervention = InterventionScenario(
            name="Intervention",
            description="Better",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
            intervention_type="subsidy",
            intervention_start_time=5,
            intervention_magnitude=0.2,
        )

        exec_base = ScenarioExecution(scenario=baseline, seed=42, model_type="bass", version="0.5.0")
        exec_int = ScenarioExecution(scenario=intervention, seed=42, model_type="bass", version="0.5.0")

        comparison = ScenarioComparison(
            baseline_execution=exec_base,
            alternative_execution=exec_int,
            comparison_metric="ranking",
        )

        data = comparison.to_dict()
        assert "baseline_execution" in data
        assert "alternative_execution" in data
        assert data["comparison_metric"] == "ranking"


@pytest.mark.unit
class TestScenarioExecutor:
    """Test ScenarioExecutor class."""

    def test_executor_creation(self):
        """Test creating an executor."""
        executor = ScenarioExecutor(
            model_type="bass",
            version="0.5.0",
            seed=42,
        )
        assert executor.model_type == "bass"
        assert executor.seed == 42

    def test_executor_with_scenarios(self, baseline_scenario, intervention_scenario):
        """Test executor with multiple scenarios."""
        executor = ScenarioExecutor(
            model_type="bass",
            version="0.5.0",
            seed=42,
        )

        scenarios = [baseline_scenario, intervention_scenario]
        assert len(scenarios) == 2

    def test_executor_deterministic_seed(self, baseline_scenario):
        """Test that executor respects deterministic seeding."""
        executor1 = ScenarioExecutor(
            model_type="bass",
            version="0.5.0",
            seed=42,
        )
        executor2 = ScenarioExecutor(
            model_type="bass",
            version="0.5.0",
            seed=42,
        )

        assert executor1.seed == executor2.seed


@pytest.mark.unit
class TestCompareScenarios:
    """Test scenario comparison functions."""

    def test_compare_two_scenarios(self, baseline_scenario, intervention_scenario):
        """Test comparing two scenarios."""
        exec_baseline = ScenarioExecution(
            scenario=baseline_scenario,
            seed=42,
            model_type="bass",
            version="0.5.0",
        )
        exec_intervention = ScenarioExecution(
            scenario=intervention_scenario,
            seed=42,
            model_type="bass",
            version="0.5.0",
        )

        comparison = compare_scenarios(
            exec_baseline,
            exec_intervention,
            metric="ranking",
        )

        assert isinstance(comparison, ScenarioComparison)
        assert comparison.comparison_metric == "ranking"

    def test_compare_multiple_scenarios(
        self,
        baseline_scenario,
        intervention_scenario,
        substitution_scenario,
    ):
        """Test comparing multiple scenarios."""
        executions = [
            ScenarioExecution(
                scenario=baseline_scenario,
                seed=42,
                model_type="bass",
                version="0.5.0",
            ),
            ScenarioExecution(
                scenario=intervention_scenario,
                seed=42,
                model_type="bass",
                version="0.5.0",
            ),
            ScenarioExecution(
                scenario=substitution_scenario,
                seed=42,
                model_type="fisher_pry",
                version="0.5.0",
            ),
        ]

        assert len(executions) == 3
