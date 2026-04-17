"""Canonical exports for advanced diffusion model workflows."""

from .advanced import AdvancedDiffusionModel, AdvancedModelSummary
from .hierarchical import HierarchicalModel
from .mixture import MixtureModel

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
    "RegimeSwitchingDiffusionModel",
]

