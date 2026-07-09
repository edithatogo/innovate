"""Agent-based and network simulation surfaces for innovate.

The supported base path is the Kairos-aligned adapter
(:class:`~innovate.abm.kairos_adapter.KairosSimulationAdapter`). Legacy
Mesa/NDLib modules require the optional ``legacy-abm`` extra and are loaded
through :mod:`innovate.abm.legacy` fail-safes.
"""

from __future__ import annotations

from innovate.abm.kairos_adapter import (
    KairosSimulationAdapter,
    bridge_crate_available,
    collect_kairos_dependency_evidence,
)
from innovate.abm.kairos_contract import (
    KAIROS_ADAPTER_SCHEMA_VERSION,
    KAIROS_PINNED_REVISION,
    KAIROS_SOURCE_URL,
    ABMBehaviorUpdate,
    AgentStateSpec,
    DESTrajectoryEvent,
    InterventionSpec,
    KairosDependencyEvidence,
    KairosSimulationRequest,
    KairosSimulationResult,
    PolicyNetworkTrace,
    RandomStreamConfig,
    SchedulerEvent,
    SimulationSeed,
    TelemetryArtifact,
    TopologySpec,
)
from innovate.abm.legacy import (
    LEGACY_ABM_EXTRA,
    LEGACY_MIGRATION_NOTE,
    LegacyABMDependencyError,
    legacy_available,
    load_legacy_module,
    migration_guidance,
    require_legacy_stack,
)

__all__ = [
    "ABMBehaviorUpdate",
    "AgentStateSpec",
    "DESTrajectoryEvent",
    "InterventionSpec",
    "KAIROS_ADAPTER_SCHEMA_VERSION",
    "KAIROS_PINNED_REVISION",
    "KAIROS_SOURCE_URL",
    "KairosDependencyEvidence",
    "KairosSimulationAdapter",
    "KairosSimulationRequest",
    "KairosSimulationResult",
    "LEGACY_ABM_EXTRA",
    "LEGACY_MIGRATION_NOTE",
    "LegacyABMDependencyError",
    "PolicyNetworkTrace",
    "RandomStreamConfig",
    "SchedulerEvent",
    "SimulationSeed",
    "TelemetryArtifact",
    "TopologySpec",
    "bridge_crate_available",
    "collect_kairos_dependency_evidence",
    "legacy_available",
    "load_legacy_module",
    "migration_guidance",
    "require_legacy_stack",
]

# Soft re-exports for environments that install innovate[legacy-abm].
try:
    from innovate.abm.agent import InnovationAgent
    from innovate.abm.competitive_diffusion import (
        CompetitiveDiffusionAgent,
        CompetitiveDiffusionModel,
    )
    from innovate.abm.disruptive_innovation import (
        DisruptiveInnovationAgent,
        DisruptiveInnovationModel,
    )
    from innovate.abm.model import InnovationModel
    from innovate.abm.sentiment_hype_cycle import SentimentHypeAgent, SentimentHypeModel

    __all__ += [
        "CompetitiveDiffusionAgent",
        "CompetitiveDiffusionModel",
        "DisruptiveInnovationAgent",
        "DisruptiveInnovationModel",
        "InnovationAgent",
        "InnovationModel",
        "SentimentHypeAgent",
        "SentimentHypeModel",
    ]
except ImportError:
    # Base install intentionally omits Mesa/NDLib.
    pass
