"""Scenario capability registry for experiment workflows.

This module defines scenario capabilities and maintains a registry of
supported scenario types for discovery, validation, and polyglot binding.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from innovate.stability import StabilityTier, normalize_stability_tier


@dataclass(frozen=True, slots=True)
class ScenarioCapability:
    """Machine-readable summary of a scenario family's public contract."""

    key: str
    scenario_type: str
    import_path: str
    stability: str = "stable"
    supports_json_payload: bool = True
    supports_arrow_payload: bool = True
    supports_rust_binding: bool = True
    supports_python_binding: bool = True

    @property
    def stability_tier(self) -> StabilityTier:
        """Return the normalized stability tier for the scenario capability."""
        return normalize_stability_tier(self.stability)


_SCENARIO_REGISTRY = MappingProxyType(
    {
        "baseline": ScenarioCapability(
            key="baseline",
            scenario_type="baseline",
            import_path="innovate.scenario.BaselineScenario",
            stability="stable",
            supports_json_payload=True,
            supports_arrow_payload=True,
            supports_rust_binding=True,
            supports_python_binding=True,
        ),
        "intervention": ScenarioCapability(
            key="intervention",
            scenario_type="intervention",
            import_path="innovate.scenario.InterventionScenario",
            stability="stable",
            supports_json_payload=True,
            supports_arrow_payload=True,
            supports_rust_binding=True,
            supports_python_binding=True,
        ),
        "substitution": ScenarioCapability(
            key="substitution",
            scenario_type="substitution",
            import_path="innovate.scenario.SubstitutionScenario",
            stability="stable",
            supports_json_payload=True,
            supports_arrow_payload=True,
            supports_rust_binding=True,
            supports_python_binding=True,
        ),
        "competition": ScenarioCapability(
            key="competition",
            scenario_type="competition",
            import_path="innovate.scenario.CompetitionScenario",
            stability="stable",
            supports_json_payload=True,
            supports_arrow_payload=True,
            supports_rust_binding=True,
            supports_python_binding=True,
        ),
        "network": ScenarioCapability(
            key="network",
            scenario_type="network",
            import_path="innovate.scenario.NetworkScenario",
            stability="stable",
            supports_json_payload=True,
            supports_arrow_payload=True,
            supports_rust_binding=True,
            supports_python_binding=True,
        ),
    },
)


def get_scenario_registry() -> Mapping[str, ScenarioCapability]:
    """Return the immutable registry for scenario families.

    Returns
    -------
    Mapping[str, ScenarioCapability]
        Immutable mapping of scenario keys to their capability metadata.
    """
    return _SCENARIO_REGISTRY


def get_scenario_capability(key: str) -> ScenarioCapability:
    """Look up one scenario family by registry key.

    Parameters
    ----------
    key
        Registry key for the scenario type (e.g., 'baseline', 'intervention').

    Returns
    -------
    ScenarioCapability
        The capability metadata for the scenario type.

    Raises
    ------
    KeyError
        If the scenario type is not registered.
    """
    try:
        return _SCENARIO_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_SCENARIO_REGISTRY))
        raise KeyError(f"Unknown scenario capability {key!r}. Available scenarios: {available}") from exc
