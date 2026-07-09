"""Fail-safe loaders and migration notes for legacy Mesa/NDLib ABM surfaces."""

from __future__ import annotations

from importlib import import_module
from typing import Any

LEGACY_ABM_EXTRA = "legacy-abm"
LEGACY_MIGRATION_NOTE = (
    "Legacy Mesa/NDLib ABM surfaces require the optional '{extra}' extra "
    "(`pip install innovate[{extra}]`). The supported migration path is the "
    "Kairos-aligned adapter in `innovate.abm.kairos_adapter` with contracts in "
    "`innovate.abm.kairos_contract`. Mesa and NDLib are not base runtime "
    "dependencies."
).format(extra=LEGACY_ABM_EXTRA)

_LEGACY_MODULES = {
    "agent": "innovate.abm.agent",
    "model": "innovate.abm.model",
    "competitive_diffusion": "innovate.abm.competitive_diffusion",
    "disruptive_innovation": "innovate.abm.disruptive_innovation",
    "sentiment_hype_cycle": "innovate.abm.sentiment_hype_cycle",
    "ndlib_model": "innovate.abm.ndlib_model",
}


class LegacyABMDependencyError(ImportError):
    """Raised when legacy ABM imports are requested without optional deps."""

    def __init__(self, missing: str, *, cause: BaseException | None = None) -> None:
        message = (
            f"Cannot import legacy ABM dependency '{missing}'. {LEGACY_MIGRATION_NOTE}"
        )
        super().__init__(message)
        self.missing = missing
        if cause is not None:
            self.__cause__ = cause


def migration_guidance() -> dict[str, str]:
    """Return structured migration guidance for docs and diagnostics."""
    return {
        "legacy_extra": LEGACY_ABM_EXTRA,
        "install": f"pip install innovate[{LEGACY_ABM_EXTRA}]",
        "replacement": "innovate.abm.kairos_adapter.KairosSimulationAdapter",
        "contract": "innovate.abm.kairos_contract.KairosSimulationRequest",
        "note": LEGACY_MIGRATION_NOTE,
    }


def legacy_available() -> dict[str, bool]:
    """Probe whether mesa/ndlib can be imported without loading ABM models."""
    status = {"mesa": False, "ndlib": False}
    for package in status:
        try:
            import_module(package)
            status[package] = True
        except ImportError:
            status[package] = False
    return status


def load_legacy_module(name: str) -> Any:
    """Import a legacy ABM module with fail-safe diagnostics.

    Parameters
    ----------
    name:
        Short module key such as ``model`` or ``ndlib_model``.
    """
    if name not in _LEGACY_MODULES:
        raise KeyError(f"Unknown legacy ABM module key: {name}")
    try:
        return import_module(_LEGACY_MODULES[name])
    except ImportError as exc:
        missing = "mesa"
        text = str(exc).lower()
        if "ndlib" in text:
            missing = "ndlib"
        elif "mesa" in text:
            missing = "mesa"
        raise LegacyABMDependencyError(missing, cause=exc) from exc


def require_legacy_stack(*, need_ndlib: bool = False) -> None:
    """Raise a clear error when the legacy optional stack is incomplete."""
    status = legacy_available()
    if not status["mesa"]:
        raise LegacyABMDependencyError("mesa")
    if need_ndlib and not status["ndlib"]:
        raise LegacyABMDependencyError("ndlib")
