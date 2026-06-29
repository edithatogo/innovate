"""Tests for scenario registry metadata.

This module tests the scenario capability registry that exposes scenario
payload families and their compatibility with bindings.
"""

import pytest

from innovate.scenario.registry import ScenarioCapability, get_scenario_registry


@pytest.mark.unit
class TestScenarioCapability:
    """Test ScenarioCapability class."""

    def test_create_scenario_capability(self):
        """Test creating a scenario capability."""
        capability = ScenarioCapability(
            key="baseline",
            scenario_type="baseline",
            import_path="innovate.scenario.BaselineScenario",
            stability="stable",
        )
        assert capability.key == "baseline"
        assert capability.scenario_type == "baseline"
        assert capability.stability == "stable"

    def test_scenario_capability_import_path(self):
        """Test that import path is correctly set."""
        capability = ScenarioCapability(
            key="intervention",
            scenario_type="intervention",
            import_path="innovate.scenario.InterventionScenario",
        )
        assert capability.import_path == "innovate.scenario.InterventionScenario"


@pytest.mark.unit
class TestScenarioRegistry:
    """Test scenario capability registry."""

    def test_get_scenario_registry(self):
        """Test retrieving the scenario registry."""
        from types import MappingProxyType

        registry = get_scenario_registry()
        assert isinstance(registry, (dict, MappingProxyType))
        assert len(registry) > 0

    def test_baseline_scenario_registered(self):
        """Test that baseline scenario is registered."""
        registry = get_scenario_registry()
        assert "baseline" in registry
        baseline = registry["baseline"]
        assert baseline.scenario_type == "baseline"
        assert baseline.stability == "stable"

    def test_intervention_scenario_registered(self):
        """Test that intervention scenario is registered."""
        registry = get_scenario_registry()
        assert "intervention" in registry
        intervention = registry["intervention"]
        assert intervention.scenario_type == "intervention"

    def test_substitution_scenario_registered(self):
        """Test that substitution scenario is registered."""
        registry = get_scenario_registry()
        assert "substitution" in registry
        substitution = registry["substitution"]
        assert substitution.scenario_type == "substitution"

    def test_competition_scenario_registered(self):
        """Test that competition scenario is registered."""
        registry = get_scenario_registry()
        assert "competition" in registry
        competition = registry["competition"]
        assert competition.scenario_type == "competition"

    def test_network_scenario_registered(self):
        """Test that network scenario is registered."""
        registry = get_scenario_registry()
        assert "network" in registry
        network = registry["network"]
        assert network.scenario_type == "network"

    def test_all_scenarios_have_compatible_payloads(self):
        """Test that all scenarios support JSON/Arrow payloads."""
        registry = get_scenario_registry()
        for key, capability in registry.items():
            assert capability.supports_json_payload
            assert capability.supports_arrow_payload

    def test_scenario_keys_match_types(self):
        """Test that registry keys match scenario types."""
        registry = get_scenario_registry()
        for key, capability in registry.items():
            assert key == capability.scenario_type
