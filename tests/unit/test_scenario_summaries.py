"""Tests for scenario comparison summaries.

This module tests ranking, incremental effect, threshold timing, and
uncertainty summarization functions.
"""

import numpy as np
import pytest

from innovate.scenario.execution import ScenarioExecution
from innovate.scenario.schemas import (
    BaselineScenario,
    InterventionScenario,
    SubstitutionScenario,
)
from innovate.scenario.summaries import (
    compute_incremental_effect,
    compute_ranking,
    compute_threshold_timing,
    compute_uncertainty,
    summarize_comparison,
)


@pytest.fixture
def baseline_execution() -> ScenarioExecution:
    """Create a baseline execution."""
    scenario = BaselineScenario(
        name="Baseline",
        description="Reference",
        time_horizon=20,
        time_unit="years",
        reference_year=2026,
        market_size=1000000,
        initial_adoption=0.01,
    )
    time_points = np.array([0, 5, 10, 15, 20], dtype=float)
    # Moderate adoption curve
    adoption = np.array([10000, 100000, 300000, 600000, 850000], dtype=float)
    return ScenarioExecution(
        scenario=scenario,
        seed=42,
        model_type="bass",
        version="0.5.0",
        time_points=time_points,
        adoption_curve=adoption,
    )


@pytest.fixture
def intervention_execution() -> ScenarioExecution:
    """Create an intervention execution."""
    scenario = InterventionScenario(
        name="Policy",
        description="With subsidy",
        time_horizon=20,
        time_unit="years",
        reference_year=2026,
        market_size=1000000,
        initial_adoption=0.01,
        intervention_type="subsidy",
        intervention_start_time=5,
        intervention_magnitude=0.2,
    )
    time_points = np.array([0, 5, 10, 15, 20], dtype=float)
    # Higher adoption curve (more aggressive)
    adoption = np.array([10000, 120000, 400000, 750000, 950000], dtype=float)
    return ScenarioExecution(
        scenario=scenario,
        seed=42,
        model_type="bass",
        version="0.5.0",
        time_points=time_points,
        adoption_curve=adoption,
    )


@pytest.mark.unit
class TestComputeRanking:
    """Test ranking computation."""

    def test_rank_single_metric(self, baseline_execution, intervention_execution):
        """Test ranking by final adoption."""
        ranking = compute_ranking(
            baseline_execution,
            intervention_execution,
            metric="final_adoption",
        )
        assert isinstance(ranking, dict)
        assert "baseline_rank" in ranking
        assert "alternative_rank" in ranking

    def test_rank_returns_numeric(self, baseline_execution, intervention_execution):
        """Test that ranking returns numeric scores."""
        ranking = compute_ranking(
            baseline_execution,
            intervention_execution,
            metric="final_adoption",
        )
        assert isinstance(ranking["baseline_rank"], (int, float))
        assert isinstance(ranking["alternative_rank"], (int, float))

    def test_higher_adoption_ranks_higher(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test that higher final adoption gets higher rank."""
        ranking = compute_ranking(
            baseline_execution,
            intervention_execution,
            metric="final_adoption",
        )
        # Intervention has higher final adoption (950k vs 850k)
        # So intervention should rank higher
        assert ranking["alternative_rank"] > ranking["baseline_rank"]


@pytest.mark.unit
class TestComputeIncrementalEffect:
    """Test incremental effect computation."""

    def test_compute_adoption_increase(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test computing adoption increase."""
        effect = compute_incremental_effect(
            baseline_execution,
            intervention_execution,
            metric="final_adoption_increase",
        )
        assert isinstance(effect, dict)
        assert "absolute_increase" in effect
        assert "relative_increase_percent" in effect

    def test_adoption_increase_positive(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test that adoption increase is positive."""
        effect = compute_incremental_effect(
            baseline_execution,
            intervention_execution,
            metric="final_adoption_increase",
        )
        # Intervention adoption (950k) > Baseline (850k)
        assert effect["absolute_increase"] > 0
        assert effect["relative_increase_percent"] > 0

    def test_adoption_increase_magnitude(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test that adoption increase magnitude is correct."""
        effect = compute_incremental_effect(
            baseline_execution,
            intervention_execution,
            metric="final_adoption_increase",
        )
        baseline_final = baseline_execution.adoption_curve[-1]
        intervention_final = intervention_execution.adoption_curve[-1]
        expected_absolute = intervention_final - baseline_final

        assert np.isclose(effect["absolute_increase"], expected_absolute, rtol=0.01)


@pytest.mark.unit
class TestComputeThresholdTiming:
    """Test threshold timing computation."""

    def test_compute_50percent_threshold(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test computing time to 50% adoption."""
        timing = compute_threshold_timing(
            baseline_execution,
            intervention_execution,
            threshold=0.5,
        )
        assert isinstance(timing, dict)
        assert "baseline_time_to_threshold" in timing
        assert "alternative_time_to_threshold" in timing

    def test_threshold_timing_returns_numeric(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test that threshold timing returns numeric values."""
        timing = compute_threshold_timing(
            baseline_execution,
            intervention_execution,
            threshold=0.3,
        )
        assert isinstance(timing["baseline_time_to_threshold"], (int, float, type(None)))
        assert isinstance(timing["alternative_time_to_threshold"], (int, float, type(None)))

    def test_faster_threshold_crossing(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test that intervention reaches threshold faster."""
        timing = compute_threshold_timing(
            baseline_execution,
            intervention_execution,
            threshold=0.5,
        )
        base_time = timing["baseline_time_to_threshold"]
        alt_time = timing["alternative_time_to_threshold"]

        # If both have values, intervention should be faster
        if base_time is not None and alt_time is not None:
            assert alt_time <= base_time


@pytest.mark.unit
class TestComputeUncertainty:
    """Test uncertainty computation."""

    def test_compute_confidence_bounds(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test computing confidence bounds."""
        uncertainty = compute_uncertainty(
            baseline_execution,
            intervention_execution,
            confidence_level=0.95,
        )
        assert isinstance(uncertainty, dict)
        assert "baseline_lower_bound" in uncertainty
        assert "baseline_upper_bound" in uncertainty
        assert "alternative_lower_bound" in uncertainty
        assert "alternative_upper_bound" in uncertainty

    def test_uncertainty_bounds_reasonable(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test that uncertainty bounds are reasonable."""
        uncertainty = compute_uncertainty(
            baseline_execution,
            intervention_execution,
            confidence_level=0.95,
        )

        # Lower bound should be below final adoption
        # Upper bound should be above final adoption
        baseline_final = baseline_execution.adoption_curve[-1]
        assert uncertainty["baseline_lower_bound"] <= baseline_final
        assert uncertainty["baseline_upper_bound"] >= baseline_final


@pytest.mark.unit
class TestSummarizeComparison:
    """Test comprehensive comparison summarization."""

    def test_summarize_returns_dict(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test that summarize returns a dict."""
        summary = summarize_comparison(
            baseline_execution,
            intervention_execution,
        )
        assert isinstance(summary, dict)

    def test_summary_contains_all_metrics(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test that summary contains all metric types."""
        summary = summarize_comparison(
            baseline_execution,
            intervention_execution,
        )
        assert "ranking" in summary
        assert "incremental_effect" in summary
        assert "threshold_timing" in summary
        assert "uncertainty" in summary

    def test_summary_is_json_serializable(
        self,
        baseline_execution,
        intervention_execution,
    ):
        """Test that summary can be serialized to JSON-compatible format."""
        import json

        summary = summarize_comparison(
            baseline_execution,
            intervention_execution,
        )

        # Convert NaN/Inf to None for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            else:
                return obj

        clean_summary = clean_for_json(summary)
        json_str = json.dumps(clean_summary)
        assert isinstance(json_str, str)
