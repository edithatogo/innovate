"""Tests for hosted-service and remote-execution documentation."""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/astro-site/src/content/docs/operations/remote-execution.md")
LATEST_DOC_PATH = Path("docs/astro-site/src/content/docs/latest/operations/remote-execution.md")
STARLIGHT_CONFIG = Path("docs/astro-site/starlight.config.mjs")


def _docs_text() -> str:
    return "\n".join((DOC_PATH.read_text(), LATEST_DOC_PATH.read_text()))


def test_remote_execution_docs_are_published_in_starlight_navigation() -> None:
    """Remote execution docs should be visible from the canonical Starlight pages."""
    starlight_config = STARLIGHT_CONFIG.read_text()

    assert DOC_PATH.is_file()
    assert LATEST_DOC_PATH.is_file()
    assert "/operations/remote-execution/" in starlight_config
    assert "slug: latest/operations/remote-execution" in LATEST_DOC_PATH.read_text()


def test_remote_execution_docs_define_contract_and_threat_model() -> None:
    """Docs should cover the hosted boundary, controls, and provenance fields."""
    docs = _docs_text()

    for phrase in (
        "RemoteExecutionRequest",
        "RemoteExecutionResponse",
        "tenant_id",
        "Authorization",
        "data-retention",
        "structured logs",
        "JAX/XLA",
        "Rust-native",
        "bridge fallback",
        "XLA internals are not a public contract",
        "InProcessRemoteExecutor",
    ):
        assert phrase in docs


def test_roadmap_mentions_remote_execution_contract_slice() -> None:
    """The roadmap should summarize the implemented remote execution slice."""
    roadmap = Path("docs/architecture_modernization_roadmap.md").read_text()

    assert "remote execution contract" in roadmap
    assert "observability" in roadmap
    assert "backend provenance" in roadmap
