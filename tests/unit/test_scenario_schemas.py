"""Tests for scenario schema models and artifact envelopes.

This module tests the scenario specification schemas for baseline, intervention,
substitution, competition, and network diffusion workflows.
"""

import json
from datetime import datetime
from typing import Any

import pytest

from innovate.scenario.schemas import (
    ArtifactEnvelope,
    BaselineScenario,
    CompetitionScenario,
    InterventionScenario,
    NetworkScenario,
    SubstitutionScenario,
)


@pytest.mark.unit
class TestBaselineScenario:
    """Test BaselineScenario schema."""

    def test_create_baseline_scenario(self):
        """Test creating a baseline scenario."""
        baseline = BaselineScenario(
            name="Baseline 2026",
            description="Reference scenario with no interventions",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
        )
        assert baseline.name == "Baseline 2026"
        assert baseline.time_horizon == 20
        assert baseline.market_size == 1000000
        assert baseline.initial_adoption == 0.01

    def test_baseline_scenario_validation_time_horizon(self):
        """Test that time_horizon must be positive."""
        with pytest.raises(ValueError):
            BaselineScenario(
                name="Invalid",
                description="Invalid baseline",
                time_horizon=-5,
                time_unit="years",
                reference_year=2026,
                market_size=1000000,
                initial_adoption=0.01,
            )

    def test_baseline_scenario_validation_market_size(self):
        """Test that market_size must be positive."""
        with pytest.raises(ValueError):
            BaselineScenario(
                name="Invalid",
                description="Invalid baseline",
                time_horizon=20,
                time_unit="years",
                reference_year=2026,
                market_size=-1000,
                initial_adoption=0.01,
            )

    def test_baseline_scenario_validation_adoption(self):
        """Test that initial_adoption must be between 0 and 1."""
        with pytest.raises(ValueError):
            BaselineScenario(
                name="Invalid",
                description="Invalid baseline",
                time_horizon=20,
                time_unit="years",
                reference_year=2026,
                market_size=1000000,
                initial_adoption=1.5,
            )

    def test_baseline_scenario_to_dict(self):
        """Test converting baseline scenario to dict."""
        baseline = BaselineScenario(
            name="Baseline",
            description="Test baseline",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
        )
        data = baseline.to_dict()
        assert data["name"] == "Baseline"
        assert data["scenario_type"] == "baseline"
        assert data["time_horizon"] == 20


@pytest.mark.unit
class TestInterventionScenario:
    """Test InterventionScenario schema."""

    def test_create_intervention_scenario(self):
        """Test creating an intervention scenario."""
        intervention = InterventionScenario(
            name="Policy Intervention A",
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
        assert intervention.name == "Policy Intervention A"
        assert intervention.intervention_type == "subsidy"
        assert intervention.intervention_magnitude == 0.2

    def test_intervention_scenario_validation_start_time(self):
        """Test that intervention_start_time must be non-negative."""
        with pytest.raises(ValueError):
            InterventionScenario(
                name="Invalid",
                description="Invalid intervention",
                time_horizon=20,
                time_unit="years",
                reference_year=2026,
                market_size=1000000,
                initial_adoption=0.01,
                intervention_type="subsidy",
                intervention_start_time=-1,
                intervention_magnitude=0.2,
            )

    def test_intervention_scenario_to_dict(self):
        """Test converting intervention scenario to dict."""
        intervention = InterventionScenario(
            name="Policy",
            description="Test intervention",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
            intervention_type="subsidy",
            intervention_start_time=5,
            intervention_magnitude=0.2,
        )
        data = intervention.to_dict()
        assert data["scenario_type"] == "intervention"
        assert data["intervention_type"] == "subsidy"


@pytest.mark.unit
class TestSubstitutionScenario:
    """Test SubstitutionScenario schema."""

    def test_create_substitution_scenario(self):
        """Test creating a substitution scenario."""
        substitution = SubstitutionScenario(
            name="Technology Substitution",
            description="Old tech replaced by new tech",
            time_horizon=25,
            time_unit="years",
            reference_year=2026,
            market_size=500000,
            initial_adoption=0.05,
            incumbent_name="OldTech",
            entrant_name="NewTech",
            substitution_rate=0.1,
        )
        assert substitution.incumbent_name == "OldTech"
        assert substitution.entrant_name == "NewTech"
        assert substitution.substitution_rate == 0.1

    def test_substitution_scenario_validation_rate(self):
        """Test that substitution_rate must be between 0 and 1."""
        with pytest.raises(ValueError):
            SubstitutionScenario(
                name="Invalid",
                description="Invalid substitution",
                time_horizon=25,
                time_unit="years",
                reference_year=2026,
                market_size=500000,
                initial_adoption=0.05,
                incumbent_name="OldTech",
                entrant_name="NewTech",
                substitution_rate=1.5,
            )

    def test_substitution_scenario_to_dict(self):
        """Test converting substitution scenario to dict."""
        substitution = SubstitutionScenario(
            name="Substitution",
            description="Test substitution",
            time_horizon=25,
            time_unit="years",
            reference_year=2026,
            market_size=500000,
            initial_adoption=0.05,
            incumbent_name="OldTech",
            entrant_name="NewTech",
            substitution_rate=0.1,
        )
        data = substitution.to_dict()
        assert data["scenario_type"] == "substitution"
        assert data["incumbent_name"] == "OldTech"


@pytest.mark.unit
class TestCompetitionScenario:
    """Test CompetitionScenario schema."""

    def test_create_competition_scenario(self):
        """Test creating a competition scenario."""
        competition = CompetitionScenario(
            name="Two-Product Market",
            description="Two products competing",
            time_horizon=30,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.02,
            num_competitors=2,
            competitor_names=["Product A", "Product B"],
            market_share_initial=[0.6, 0.4],
        )
        assert competition.num_competitors == 2
        assert len(competition.competitor_names) == 2
        assert competition.market_share_initial == [0.6, 0.4]

    def test_competition_scenario_validation_market_share(self):
        """Test that market shares must sum to 1.0."""
        with pytest.raises(ValueError):
            CompetitionScenario(
                name="Invalid",
                description="Invalid competition",
                time_horizon=30,
                time_unit="years",
                reference_year=2026,
                market_size=1000000,
                initial_adoption=0.02,
                num_competitors=2,
                competitor_names=["Product A", "Product B"],
                market_share_initial=[0.4, 0.4],  # Sum to 0.8, not 1.0
            )

    def test_competition_scenario_to_dict(self):
        """Test converting competition scenario to dict."""
        competition = CompetitionScenario(
            name="Competition",
            description="Test competition",
            time_horizon=30,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.02,
            num_competitors=2,
            competitor_names=["Product A", "Product B"],
            market_share_initial=[0.6, 0.4],
        )
        data = competition.to_dict()
        assert data["scenario_type"] == "competition"
        assert data["num_competitors"] == 2


@pytest.mark.unit
class TestNetworkScenario:
    """Test NetworkScenario schema."""

    def test_create_network_scenario(self):
        """Test creating a network scenario."""
        network = NetworkScenario(
            name="Network Diffusion",
            description="Adoption through social networks",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=100000,
            initial_adoption=0.01,
            network_type="scale_free",
            num_nodes=1000,
            average_degree=5,
        )
        assert network.network_type == "scale_free"
        assert network.num_nodes == 1000
        assert network.average_degree == 5

    def test_network_scenario_validation_nodes(self):
        """Test that num_nodes must be positive."""
        with pytest.raises(ValueError):
            NetworkScenario(
                name="Invalid",
                description="Invalid network",
                time_horizon=20,
                time_unit="years",
                reference_year=2026,
                market_size=100000,
                initial_adoption=0.01,
                network_type="scale_free",
                num_nodes=-100,
                average_degree=5,
            )

    def test_network_scenario_to_dict(self):
        """Test converting network scenario to dict."""
        network = NetworkScenario(
            name="Network",
            description="Test network",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=100000,
            initial_adoption=0.01,
            network_type="scale_free",
            num_nodes=1000,
            average_degree=5,
        )
        data = network.to_dict()
        assert data["scenario_type"] == "network"
        assert data["network_type"] == "scale_free"


@pytest.mark.unit
class TestArtifactEnvelope:
    """Test ArtifactEnvelope schema."""

    def test_create_artifact_envelope(self):
        """Test creating an artifact envelope."""
        baseline = BaselineScenario(
            name="Test",
            description="Test baseline",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
        )

        envelope = ArtifactEnvelope(
            scenario=baseline,
            seed=42,
            model_type="bass",
            version="0.5.0",
            notes="Test run",
        )

        assert envelope.scenario.name == "Test"
        assert envelope.seed == 42
        assert envelope.model_type == "bass"
        assert envelope.version == "0.5.0"

    def test_artifact_envelope_timestamp(self):
        """Test that artifact envelope has timestamp."""
        baseline = BaselineScenario(
            name="Test",
            description="Test baseline",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
        )

        envelope = ArtifactEnvelope(
            scenario=baseline,
            seed=42,
            model_type="bass",
            version="0.5.0",
        )

        assert envelope.timestamp is not None
        assert isinstance(envelope.timestamp, datetime)

    def test_artifact_envelope_to_dict(self):
        """Test converting artifact envelope to dict."""
        baseline = BaselineScenario(
            name="Test",
            description="Test baseline",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
        )

        envelope = ArtifactEnvelope(
            scenario=baseline,
            seed=42,
            model_type="bass",
            version="0.5.0",
            notes="Test",
        )

        data = envelope.to_dict()
        assert data["seed"] == 42
        assert data["model_type"] == "bass"
        assert data["version"] == "0.5.0"

    def test_artifact_envelope_json_serializable(self):
        """Test that artifact envelope is JSON serializable."""
        baseline = BaselineScenario(
            name="Test",
            description="Test baseline",
            time_horizon=20,
            time_unit="years",
            reference_year=2026,
            market_size=1000000,
            initial_adoption=0.01,
        )

        envelope = ArtifactEnvelope(
            scenario=baseline,
            seed=42,
            model_type="bass",
            version="0.5.0",
        )

        json_str = json.dumps(envelope.to_dict())
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["seed"] == 42
