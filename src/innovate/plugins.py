"""Extension manifest validation and in-memory registration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .stability import StabilityTier, normalize_stability_tier

_KNOWN_EXTENSION_POINTS = {
    "dataset_provider",
    "diagnostics",
    "model_registry",
    "serialization_adapter",
}

_REGISTERED_EXTENSIONS: dict[str, ExtensionManifest] = {}


@dataclass(frozen=True, slots=True)
class ExtensionManifest:
    """Canonical manifest for a local extension or plugin integration."""

    name: str
    version: str
    entrypoint: str
    stability: StabilityTier | str
    extension_points: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""
    requires: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize and validate the manifest."""
        object.__setattr__(self, "stability", normalize_stability_tier(self.stability))
        object.__setattr__(self, "extension_points", tuple(self.extension_points))
        object.__setattr__(self, "requires", tuple(self.requires))

        if not self.name:
            raise ValueError("Extension manifest name must be non-empty.")
        if not self.version:
            raise ValueError("Extension manifest version must be non-empty.")
        if not self.entrypoint or ":" not in self.entrypoint:
            raise ValueError("Extension manifest entrypoint must be in 'module:callable' format.")
        if not self.extension_points:
            raise ValueError("Extension manifest extension_points must be non-empty.")

        unknown_points = sorted(set(self.extension_points) - _KNOWN_EXTENSION_POINTS)
        if unknown_points:
            raise ValueError(f"Unknown extension points: {', '.join(unknown_points)}.")

    def validate(self) -> None:
        """Re-run validation explicitly."""
        self.__post_init__()


def register_extension(manifest: ExtensionManifest) -> None:
    """Register a validated manifest in the local extension registry."""
    manifest.validate()
    if manifest.name in _REGISTERED_EXTENSIONS:
        raise ValueError(f"Extension {manifest.name!r} is already registered.")
    _REGISTERED_EXTENSIONS[manifest.name] = manifest


def get_registered_extensions() -> Mapping[str, ExtensionManifest]:
    """Return the immutable registry of known extension manifests."""
    return MappingProxyType(_REGISTERED_EXTENSIONS)
