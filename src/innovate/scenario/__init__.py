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
    "ArtifactEnvelope",
    # Schemas
    "BaselineScenario",
    "CompetitionScenario",
    "InterventionScenario",
    "NetworkScenario",
    # Registry
    "ScenarioCapability",
    "ScenarioComparison",
    # Execution
    "ScenarioExecution",
    "ScenarioExecutor",
    "SubstitutionScenario",
    "compare_scenarios",
    "compute_incremental_effect",
    # Summaries
    "compute_ranking",
    "compute_threshold_timing",
    "compute_uncertainty",
    "get_scenario_capability",
    "get_scenario_registry",
    "summarize_comparison",
]
