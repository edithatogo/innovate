"""Tests for the ecosystem module incubation documentation."""

from __future__ import annotations

from pathlib import Path


def test_ecosystem_module_incubation_docs_are_present() -> None:
    """The ecosystem strategy and contract outline should exist."""
    assert Path("docs/ecosystem/module_incubation_strategy.md").is_file()
    assert Path("specs/ecosystem/README.md").is_file()


def test_ecosystem_module_incubation_docs_define_roles_and_non_goals() -> None:
    """The docs should name ecosystem roles and the intended exclusions."""
    strategy = Path("docs/ecosystem/module_incubation_strategy.md").read_text()
    contract = Path("specs/ecosystem/README.md").read_text()

    for token in ("innovate", "lifecourse", "voiage", "mars", "HEOML"):
        assert token in strategy
        assert token in contract

    assert "Non-Goals" in strategy
    assert "health-economic simulation engines" in strategy
    assert "VOI methods" in strategy
    assert "private sibling-project internals" in strategy

    assert "Non-goals" in contract
    assert "pickle" in contract
    assert "mars core API" in contract
    assert "health-economic simulation or VOI engine" in contract


def test_ecosystem_module_incubation_docs_define_artifacts_and_heoml_boundary() -> None:
    """The docs should spell out the artifact groups and the HEOML boundary."""
    strategy = Path("docs/ecosystem/module_incubation_strategy.md").read_text()
    contract = Path("specs/ecosystem/README.md").read_text()

    for token in (
        "adoption_curve",
        "uptake_trajectory",
        "policy_spread_trace",
        "network_diffusion_trace",
        "diagnostics_record",
        "provenance_record",
        "Arrow or Parquet",
        "HEOML artifacts are wrappers",
        "portable base layer",
        "heoml.extensions.innovate",
        "Namespace Rule",
    ):
        assert token in strategy or token in contract


def test_ecosystem_module_incubation_docs_define_dependency_policy_and_promotion() -> None:
    """The docs should spell out the adapter gating and promotion ladder."""
    strategy = Path("docs/ecosystem/module_incubation_strategy.md").read_text()
    contract = Path("specs/ecosystem/README.md").read_text()

    for token in (
        "optional adapter",
        "deterministic smoke fixture",
        "compatibility matrix",
        "documented",
        "experimental",
        "supported",
        "smoke CI",
        "Renovate",
        "security checks",
        "removal path",
    ):
        assert token in strategy

    for token in (
        "explicit promotion stages",
        "smoke CI",
        "Renovate",
        "security",
        "compatibility matrix",
        "optional extras",
    ):
        assert token in contract


def test_ecosystem_module_incubation_docs_update_planning_files() -> None:
    """The governance docs should reference the ecosystem policy work."""
    tracks = Path("conductor/tracks.md").read_text()
    todo = Path("documents/todo.md").read_text()
    changelog = Path("CHANGELOG.md").read_text()

    assert "adapter promotion policy" in tracks
    assert "optional extras, smoke CI, and compatibility matrices" in todo
    assert "documented to experimental to supported" in todo
    assert "ecosystem dependency and promotion policy notes" in changelog
