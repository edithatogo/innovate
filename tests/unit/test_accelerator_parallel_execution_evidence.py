"""Tests for accelerator and parallel execution evidence policy."""

from __future__ import annotations

import json
from pathlib import Path

DOC_PATH = Path("docs/astro-site/src/content/docs/operations/accelerator-parallel-execution.md")
LATEST_DOC_PATH = Path("docs/astro-site/src/content/docs/latest/operations/accelerator-parallel-execution.md")
ARTIFACT_PATH = Path("docs/source/_static/accelerator_parallel_execution_evidence_schema.json")
STARLIGHT_CONFIG = Path("docs/astro-site/starlight.config.mjs")


def _doc_text() -> str:
    return "\n".join((DOC_PATH.read_text(), LATEST_DOC_PATH.read_text()))


def test_accelerator_parallel_execution_evidence_doc_covers_execution_modes() -> None:
    """The evidence page should cover accelerator, distributed, and scheduler modes."""
    doc = _doc_text()

    for phrase in (
        "CPU parallelism",
        "GPU",
        "TPU",
        "ASIC-oriented",
        "vendor-specific accelerator",
        "distributed execution",
        "scheduler-aware benchmarking",
        "Slurm",
        "PBS",
    ):
        assert phrase in doc


def test_accelerator_evidence_schema_is_backend_neutral() -> None:
    """Machine-readable evidence should keep backend internals out of public ABI."""
    schema = json.loads(ARTIFACT_PATH.read_text())

    assert schema["schema_version"] == 1
    assert schema["abi_policy"]["public_contract"] == "kernel_schema"
    assert set(schema["execution_modes"]) >= {"cpu", "gpu", "tpu", "distributed", "scheduler_aware"}
    assert "xla_lowering" in schema["abi_policy"]["forbidden_public_fields"]
    assert "jaxlib_internal" in schema["abi_policy"]["forbidden_public_fields"]
    assert "scheduler_internal_job_id" in schema["abi_policy"]["forbidden_public_fields"]


def test_evidence_schema_requires_scheduler_and_fallback_fields() -> None:
    """Artifacts should record scheduler context, fallback, and backend-neutral evidence links."""
    schema = json.loads(ARTIFACT_PATH.read_text())
    required = set(schema["required_artifact_fields"])

    assert {
        "execution_mode",
        "accelerator_target",
        "scheduler",
        "runner_command",
        "compile_time_seconds",
        "steady_state_runtime_seconds",
        "memory_observation",
        "fallback_status",
        "rejection_rationale",
        "evidence_uri",
    }.issubset(required)


def test_accelerator_evidence_docs_are_linked_from_starlight_sidebar() -> None:
    """The evidence page should be reachable from Starlight navigation."""
    starlight_config = STARLIGHT_CONFIG.read_text()

    assert DOC_PATH.is_file()
    assert LATEST_DOC_PATH.is_file()
    assert "/operations/accelerator-parallel-execution/" in starlight_config
    assert "slug: latest/operations/accelerator-parallel-execution" in LATEST_DOC_PATH.read_text()
