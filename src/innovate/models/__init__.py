"""Canonical exports for advanced diffusion model workflows."""

from .advanced import AdvancedDiffusionModel, AdvancedModelSummary
from .contracts import NetworkDiffusionInputs, PolicyTimingInputs
from .hierarchical import HierarchicalModel
from .mixture import MixtureModel
from .network import NetworkDiffusionModel
from .policy import PolicyHazardDiffusionModel

try:
    from .advanced import LatentProcessDiffusionModel, RegimeSwitchingDiffusionModel
except ImportError:  # pragma: no cover - defensive guard for partial installs
    LatentProcessDiffusionModel = None
    RegimeSwitchingDiffusionModel = None

__all__ = [
    "AdvancedDiffusionModel",
    "AdvancedModelSummary",
    "HierarchicalModel",
    "LatentProcessDiffusionModel",
    "MixtureModel",
    "NetworkDiffusionInputs",
    "NetworkDiffusionModel",
    "PolicyHazardDiffusionModel",
    "PolicyTimingInputs",
    "RegimeSwitchingDiffusionModel",
]
