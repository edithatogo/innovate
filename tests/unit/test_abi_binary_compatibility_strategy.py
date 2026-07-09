"""Static gates for the ABI and binary compatibility strategy."""

from __future__ import annotations

import json
from pathlib import Path

DOC_PATH = Path("docs/astro-site/src/content/docs/operations/abi-compatibility.md")
LATEST_DOC_PATH = Path("docs/astro-site/src/content/docs/latest/operations/abi-compatibility.md")
STARLIGHT_CONFIG = Path("docs/astro-site/starlight.config.mjs")
TRACK_DIR = Path("conductor/archive/abi_binary_compatibility_strategy_20260507")
ARCHIVE_DIR = Path("conductor/archive/abi_binary_compatibility_strategy_20260507")
REGISTRY_PATH = Path("conductor/tracks.md")


def _strategy_doc() -> str:
    return "\n".join((DOC_PATH.read_text(), LATEST_DOC_PATH.read_text()))


def _normalized_doc() -> str:
    return " ".join(_strategy_doc().split())


def test_abi_strategy_doc_is_in_starlight_navigation() -> None:
    """The ABI strategy must be discoverable from Starlight navigation."""
    assert DOC_PATH.is_file()
    assert LATEST_DOC_PATH.is_file()
    starlight_config = STARLIGHT_CONFIG.read_text()

    assert "/operations/abi-compatibility/" in starlight_config
    assert "slug: latest/operations/abi-compatibility" in LATEST_DOC_PATH.read_text()


def test_abi_strategy_separates_api_schema_and_native_abi() -> None:
    """Compatibility policy should not collapse public API, schema, and ABI."""
    doc = _normalized_doc()

    for phrase in (
        "Public API compatibility",
        "Kernel schema compatibility",
        "Native ABI compatibility",
        "API-preserving ABI changes",
        "must not require callers to import Rust structs",
        "must not require callers to link against private native symbols",
        "schema-versioned kernel request and response payloads",
        "capability-discovery metadata",
    ):
        assert phrase in doc


def test_arrow_c_data_interface_is_the_native_interchange_boundary() -> None:
    """Binary interchange should be Arrow-owned, not ad hoc native structs."""
    doc = _normalized_doc()

    for phrase in (
        "Arrow C Data Interface",
        "Arrow C Stream Interface",
        "FFI boundary for tabular arrays",
        "Arrow schema and array metadata",
        "Rust private structs are not public ABI",
        "Python objects are not public ABI",
    ):
        assert phrase in doc


def test_xla_internals_are_excluded_from_public_abi() -> None:
    """XLA implementation details should stay behind capability gates."""
    doc = _normalized_doc()

    for phrase in (
        "XLA internals are not public ABI",
        "jaxlib",
        "HLO",
        "compiled executable",
        "device buffer layout",
        "optional accelerator backend",
        "capability-gated implementation detail",
    ):
        assert phrase in doc


def test_package_manager_binary_compatibility_notes_cover_binding_targets() -> None:
    """Package-manager guidance should name the binary compatibility duties."""
    doc = _normalized_doc()

    for phrase in (
        "PyPI wheels",
        "conda packages",
        "crates.io",
        "npm",
        "CRAN",
        "R-universe",
        "Julia General",
        "Go modules",
        "NuGet",
        "manylinux",
        "musllinux",
        "macOS universal2",
        "Windows wheel tags",
    ):
        assert phrase in doc


def test_abi_track_metadata_and_registry_state_are_consistent() -> None:
    """The active or archived Conductor record should match its registry state."""
    registry = REGISTRY_PATH.read_text()
    track_dir = TRACK_DIR if TRACK_DIR.exists() else ARCHIVE_DIR
    metadata = json.loads((track_dir / "metadata.json").read_text())

    assert metadata["track_id"] == "abi_binary_compatibility_strategy_20260507"
    assert metadata["type"] == "chore"
    assert metadata["status"] in {"in_progress", "completed"}
    if metadata["status"] == "completed":
        assert track_dir == ARCHIVE_DIR
        assert "./archive/abi_binary_compatibility_strategy_20260507/" in registry
    else:
        assert track_dir == TRACK_DIR
        assert "./tracks/abi_binary_compatibility_strategy_20260507/" in registry
