"""Scenario experiment workflows module.

This module provides a first-class scenario experiment layer that lets users
define, run, compare, and export reproducible innovation, policy, substitution,
and competition scenarios.
"""

from innovate.scenario.execution import (
    ScenarioComparison,
    ScenarioExecution,
    ScenarioExecutor,
    compare_scenarios,
)
from innovate.scenario.registry import (
    ScenarioCapability,
    get_scenario_capability,
    get_scenario_registry,
)
from innovate.scenario.schemas import (
    ArtifactEnvelope,
    BaselineScenario,
    CompetitionScenario,
    InterventionScenario,
    NetworkScenario,
    SubstitutionScenario,
)
from innovate.scenario.summaries import (
    compute_incremental_effect,
    compute_ranking,
    compute_threshold_timing,
    compute_uncertainty,
    summarize_comparison,
)

__all__ = [
    # Schemas
    "BaselineScenario",
    "InterventionScenario",
    "SubstitutionScenario",
    "CompetitionScenario",
    "NetworkScenario",
    "ArtifactEnvelope",
    # Execution
    "ScenarioExecution",
    "ScenarioComparison",
    "ScenarioExecutor",
    "compare_scenarios",
    # Registry
    "ScenarioCapability",
    "get_scenario_registry",
    "get_scenario_capability",
    # Summaries
    "compute_ranking",
    "compute_incremental_effect",
    "compute_threshold_timing",
    "compute_uncertainty",
    "summarize_comparison",
]
