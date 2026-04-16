"""Canonical exports for stable competition models.

The matrix-form ``competition.MultiProductDiffusionModel`` remains the stable
user-facing API. The more configurable ``multi_product`` module stays available
as a separate implementation, but is not yet the canonical package export.
"""

from .competition import MultiProductDiffusionModel
from .lotka_volterra import LotkaVolterraModel

__all__ = [
    "LotkaVolterraModel",
    "MultiProductDiffusionModel",
]
