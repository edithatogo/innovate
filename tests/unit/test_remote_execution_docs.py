"""Tests for hosted-service and remote-execution documentation."""

from __future__ import annotations

from pathlib import Path


def test_remote_execution_docs_are_published_in_sphinx_navigation() -> None:
    """Remote execution docs should be visible from the canonical docs pages."""
    docs_root = Path("docs/source")
    index = (docs_root / "index.rst").read_text()
    package_docs = (docs_root / "innovate.rst").read_text()

    assert (docs_root / "remote_execution.rst").is_file()
    assert "remote_execution" in index
    assert "docs/source/remote_execution.rst" in package_docs


def test_remote_execution_docs_define_contract_and_threat_model() -> None:
    """Docs should cover the hosted boundary, controls, and provenance fields."""
    docs = Path("docs/source/remote_execution.rst").read_text()

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
