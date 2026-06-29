"""Scenario specification schemas for reproducible experiment workflows.

This module defines validated scenario schemas for baseline, intervention,
substitution, competition, and network diffusion scenarios.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from innovate.utils.validation import (
    validate_float,
    validate_positive_numeric_sequence,
)


class ScenarioBase(ABC):
    """Abstract base class for all scenario types."""

    def __init__(
        self,
        name: str,
        description: str,
        time_horizon: float,
        time_unit: str,
        reference_year: int,
        market_size: float,
        initial_adoption: float,
    ):
        """Initialize base scenario.

        Parameters
        ----------
        name
            Human-readable scenario name.
        description
            Detailed description of the scenario.
        time_horizon
            Time horizon in time_unit (must be positive).
        time_unit
            Unit of time (e.g., 'years', 'months', 'quarters').
        reference_year
            Reference year for the scenario (e.g., 2026).
        market_size
            Total addressable market size (must be positive).
        initial_adoption
            Initial adoption rate as fraction [0, 1].

        Raises
        ------
        ValueError
            If any parameter validation fails.
        """
        self.name = str(name)
        self.description = str(description)
        self.time_horizon = validate_float(time_horizon, "time_horizon", min_val=0)
        if self.time_horizon <= 0:
            raise ValueError("time_horizon must be positive")
        self.time_unit = str(time_unit)
        self.reference_year = int(reference_year)
        self.market_size = validate_float(market_size, "market_size", min_val=0)
        if self.market_size <= 0:
            raise ValueError("market_size must be positive")
        self.initial_adoption = validate_float(initial_adoption, "initial_adoption", min_val=0, max_val=1)
        if self.initial_adoption < 0 or self.initial_adoption > 1:
            raise ValueError("initial_adoption must be between 0 and 1")

    @property
    @abstractmethod
    def scenario_type(self) -> str:
        """Return the scenario type identifier."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert scenario to a dictionary representation."""


class BaselineScenario(ScenarioBase):
    """Baseline (reference) scenario with no interventions."""

    @property
    def scenario_type(self) -> str:
        """Return the scenario type identifier."""
        return "baseline"

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario to a dictionary representation."""
        return {
            "scenario_type": self.scenario_type,
            "name": self.name,
            "description": self.description,
            "time_horizon": self.time_horizon,
            "time_unit": self.time_unit,
            "reference_year": self.reference_year,
            "market_size": self.market_size,
            "initial_adoption": self.initial_adoption,
        }


class InterventionScenario(ScenarioBase):
    """Intervention scenario with policy or program changes."""

    def __init__(
        self,
        name: str,
        description: str,
        time_horizon: float,
        time_unit: str,
        reference_year: int,
        market_size: float,
        initial_adoption: float,
        intervention_type: str,
        intervention_start_time: float,
        intervention_magnitude: float,
    ):
        """Initialize intervention scenario.

        Parameters
        ----------
        name
            Human-readable scenario name.
        description
            Detailed description of the scenario.
        time_horizon
            Time horizon in time_unit.
        time_unit
            Unit of time.
        reference_year
            Reference year for the scenario.
        market_size
            Total addressable market size.
        initial_adoption
            Initial adoption rate as fraction [0, 1].
        intervention_type
            Type of intervention (e.g., 'subsidy', 'mandate', 'tax').
        intervention_start_time
            Time (in time_unit) when intervention begins (must be non-negative).
        intervention_magnitude
            Magnitude of intervention effect (typically [0, 1] or [0, ∞)).

        Raises
        ------
        ValueError
            If any parameter validation fails.
        """
        super().__init__(
            name,
            description,
            time_horizon,
            time_unit,
            reference_year,
            market_size,
            initial_adoption,
        )
        self.intervention_type = str(intervention_type)
        self.intervention_start_time = validate_float(intervention_start_time, "intervention_start_time", min_val=0)
        if self.intervention_start_time < 0:
            raise ValueError("intervention_start_time must be non-negative")
        self.intervention_magnitude = float(intervention_magnitude)

    @property
    def scenario_type(self) -> str:
        """Return the scenario type identifier."""
        return "intervention"

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario to a dictionary representation."""
        return {
            "scenario_type": self.scenario_type,
            "name": self.name,
            "description": self.description,
            "time_horizon": self.time_horizon,
            "time_unit": self.time_unit,
            "reference_year": self.reference_year,
            "market_size": self.market_size,
            "initial_adoption": self.initial_adoption,
            "intervention_type": self.intervention_type,
            "intervention_start_time": self.intervention_start_time,
            "intervention_magnitude": self.intervention_magnitude,
        }


class SubstitutionScenario(ScenarioBase):
    """Technology substitution scenario."""

    def __init__(
        self,
        name: str,
        description: str,
        time_horizon: float,
        time_unit: str,
        reference_year: int,
        market_size: float,
        initial_adoption: float,
        incumbent_name: str,
        entrant_name: str,
        substitution_rate: float,
    ):
        """Initialize substitution scenario.

        Parameters
        ----------
        name
            Human-readable scenario name.
        description
            Detailed description of the scenario.
        time_horizon
            Time horizon in time_unit.
        time_unit
            Unit of time.
        reference_year
            Reference year for the scenario.
        market_size
            Total addressable market size.
        initial_adoption
            Initial adoption rate as fraction [0, 1].
        incumbent_name
            Name of the incumbent technology.
        entrant_name
            Name of the entrant technology.
        substitution_rate
            Rate at which entrant replaces incumbent [0, 1].

        Raises
        ------
        ValueError
            If any parameter validation fails.
        """
        super().__init__(
            name,
            description,
            time_horizon,
            time_unit,
            reference_year,
            market_size,
            initial_adoption,
        )
        self.incumbent_name = str(incumbent_name)
        self.entrant_name = str(entrant_name)
        self.substitution_rate = validate_float(substitution_rate, "substitution_rate", min_val=0, max_val=1)
        if self.substitution_rate < 0 or self.substitution_rate > 1:
            raise ValueError("substitution_rate must be between 0 and 1")

    @property
    def scenario_type(self) -> str:
        """Return the scenario type identifier."""
        return "substitution"

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario to a dictionary representation."""
        return {
            "scenario_type": self.scenario_type,
            "name": self.name,
            "description": self.description,
            "time_horizon": self.time_horizon,
            "time_unit": self.time_unit,
            "reference_year": self.reference_year,
            "market_size": self.market_size,
            "initial_adoption": self.initial_adoption,
            "incumbent_name": self.incumbent_name,
            "entrant_name": self.entrant_name,
            "substitution_rate": self.substitution_rate,
        }


class CompetitionScenario(ScenarioBase):
    """Multi-product competition scenario."""

    def __init__(
        self,
        name: str,
        description: str,
        time_horizon: float,
        time_unit: str,
        reference_year: int,
        market_size: float,
        initial_adoption: float,
        num_competitors: int,
        competitor_names: list[str],
        market_share_initial: list[float],
    ):
        """Initialize competition scenario.

        Parameters
        ----------
        name
            Human-readable scenario name.
        description
            Detailed description of the scenario.
        time_horizon
            Time horizon in time_unit.
        time_unit
            Unit of time.
        reference_year
            Reference year for the scenario.
        market_size
            Total addressable market size.
        initial_adoption
            Initial adoption rate as fraction [0, 1].
        num_competitors
            Number of competing products.
        competitor_names
            List of competitor names (length must equal num_competitors).
        market_share_initial
            Initial market share for each competitor (must sum to 1.0).

        Raises
        ------
        ValueError
            If any parameter validation fails.
        """
        super().__init__(
            name,
            description,
            time_horizon,
            time_unit,
            reference_year,
            market_size,
            initial_adoption,
        )
        self.num_competitors = int(num_competitors)
        self.competitor_names = list(competitor_names)
        self.market_share_initial = list(market_share_initial)

        if len(self.competitor_names) != self.num_competitors:
            raise ValueError(
                f"Length of competitor_names ({len(self.competitor_names)}) "
                f"must equal num_competitors ({self.num_competitors})"
            )
        if len(self.market_share_initial) != self.num_competitors:
            raise ValueError(
                f"Length of market_share_initial ({len(self.market_share_initial)}) "
                f"must equal num_competitors ({self.num_competitors})"
            )

        # Validate market shares sum to approximately 1.0
        share_sum = sum(self.market_share_initial)
        if abs(share_sum - 1.0) > 1e-6:
            raise ValueError(f"market_share_initial must sum to 1.0, got {share_sum}")

    @property
    def scenario_type(self) -> str:
        """Return the scenario type identifier."""
        return "competition"

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario to a dictionary representation."""
        return {
            "scenario_type": self.scenario_type,
            "name": self.name,
            "description": self.description,
            "time_horizon": self.time_horizon,
            "time_unit": self.time_unit,
            "reference_year": self.reference_year,
            "market_size": self.market_size,
            "initial_adoption": self.initial_adoption,
            "num_competitors": self.num_competitors,
            "competitor_names": self.competitor_names,
            "market_share_initial": self.market_share_initial,
        }


class NetworkScenario(ScenarioBase):
    """Network diffusion scenario."""

    def __init__(
        self,
        name: str,
        description: str,
        time_horizon: float,
        time_unit: str,
        reference_year: int,
        market_size: float,
        initial_adoption: float,
        network_type: str,
        num_nodes: int,
        average_degree: float,
    ):
        """Initialize network scenario.

        Parameters
        ----------
        name
            Human-readable scenario name.
        description
            Detailed description of the scenario.
        time_horizon
            Time horizon in time_unit.
        time_unit
            Unit of time.
        reference_year
            Reference year for the scenario.
        market_size
            Total addressable market size.
        initial_adoption
            Initial adoption rate as fraction [0, 1].
        network_type
            Type of network topology (e.g., 'scale_free', 'small_world', 'random').
        num_nodes
            Number of nodes in the network (must be positive).
        average_degree
            Average degree of network nodes (must be positive).

        Raises
        ------
        ValueError
            If any parameter validation fails.
        """
        super().__init__(
            name,
            description,
            time_horizon,
            time_unit,
            reference_year,
            market_size,
            initial_adoption,
        )
        self.network_type = str(network_type)
        self.num_nodes = int(num_nodes)
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        self.average_degree = validate_float(average_degree, "average_degree", min_val=0)
        if self.average_degree <= 0:
            raise ValueError("average_degree must be positive")

    @property
    def scenario_type(self) -> str:
        """Return the scenario type identifier."""
        return "network"

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario to a dictionary representation."""
        return {
            "scenario_type": self.scenario_type,
            "name": self.name,
            "description": self.description,
            "time_horizon": self.time_horizon,
            "time_unit": self.time_unit,
            "reference_year": self.reference_year,
            "market_size": self.market_size,
            "initial_adoption": self.initial_adoption,
            "network_type": self.network_type,
            "num_nodes": self.num_nodes,
            "average_degree": self.average_degree,
        }


class ArtifactEnvelope:
    """Container for scenario execution artifacts with metadata."""

    def __init__(
        self,
        scenario: ScenarioBase,
        seed: int,
        model_type: str,
        version: str,
        notes: str | None = None,
        timestamp: datetime | None = None,
    ):
        """Initialize artifact envelope.

        Parameters
        ----------
        scenario
            The scenario specification.
        seed
            Random seed for reproducibility.
        model_type
            Type of model used (e.g., 'bass', 'fisher_pry').
        version
            Version string of the implementation.
        notes
            Optional notes about the artifact.
        timestamp
            Timestamp of creation (defaults to current time).
        """
        self.scenario = scenario
        self.seed = int(seed)
        self.model_type = str(model_type)
        self.version = str(version)
        self.notes = notes if notes is None else str(notes)
        self.timestamp = timestamp if timestamp is not None else datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert artifact envelope to a dictionary representation."""
        return {
            "scenario": self.scenario.to_dict(),
            "seed": self.seed,
            "model_type": self.model_type,
            "version": self.version,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }
