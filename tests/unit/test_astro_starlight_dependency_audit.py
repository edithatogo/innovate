"""Tests for the Astro/Starlight pre-migration dependency audit."""

from __future__ import annotations

import json
from pathlib import Path


AUDIT_PATH = Path("docs/source/_static/astro_starlight/current_dependency_audit.json")
PACKAGE_PATH = Path("docs/astro-site/package.json")
LOCKFILE_PATH = Path("docs/astro-site/pnpm-lock.yaml")
MANIFEST_PATH = Path("docs/source/_static/astro_starlight/migration_manifest.json")


def test_audit_evidence_file_exists() -> None:
    """The pre-migration dependency audit must exist."""
    assert AUDIT_PATH.exists()


def test_audit_records_package_manager() -> None:
    """The audit must record the package manager tool and version."""
    audit = json.loads(AUDIT_PATH.read_text())
    pm = audit["package_manager"]
    assert pm["tool"] == "pnpm"
    assert pm["version"] == "9.15.9"
    assert pm["lockfile_version"] == "9.0"


def test_audit_records_astro_dependency() -> None:
    """The audit must record the Astro version from both sources."""
    audit = json.loads(AUDIT_PATH.read_text())
    dep = audit["dependencies"]["astro"]
    assert dep["package_json_specifier"] == "^7.0.2"
    assert dep["lockfile_specifier"] == "^7.0.2"
    assert dep["lockfile_resolved"] == "7.0.2"


def test_audit_records_starlight_mismatch() -> None:
    """The audit must document the Starlight version discrepancy between sources."""
    audit = json.loads(AUDIT_PATH.read_text())
    dep = audit["dependencies"]["@astrojs/starlight"]
    assert dep["package_json_specifier"] == "^0.40.0"
    assert dep["lockfile_specifier"] == "^0.41.0"
    assert dep["lockfile_resolved"] == "0.41.0"
    assert "MISMATCH" in dep["note"]


def test_audit_records_all_plugin_dependencies() -> None:
    """The audit must list all active Starlight plugin packages."""
    audit = json.loads(AUDIT_PATH.read_text())
    plugins = audit["plugin_relationships"]
    assert "starlightLinksValidator" in plugins
    assert "starlightDocSearch" in plugins
    assert "starlightVersions" in plugins
    assert "polyglot" in plugins


def test_audit_records_migration_manifest_baseline() -> None:
    """The audit must reference the current migration manifest baseline for reconciliation."""
    audit = json.loads(AUDIT_PATH.read_text())
    manifest = audit["migration_manifest_baseline"]
    assert manifest["starlight"] == "0.40.0"


TARGET_DECISION_PATH = Path("docs/source/_static/astro_starlight/starlight_target_decision.json")


def test_target_decision_artifact_exists() -> None:
    """The Starlight target decision artifact must exist."""
    assert TARGET_DECISION_PATH.exists()


def test_target_decision_selects_starlight_41() -> None:
    """The decision must select Starlight 0.41.x based on lockfile evidence."""
    decision = json.loads(TARGET_DECISION_PATH.read_text())
    assert decision["selected_target"] == "Starlight 0.41.x"
    assert decision["evidence"]["lockfile_resolved_version"] == "0.41.0"


def test_target_decision_lists_actions() -> None:
    """The decision must list concrete taken and remaining actions."""
    decision = json.loads(TARGET_DECISION_PATH.read_text())
    taken = decision["actions_taken"]
    remaining = decision["actions_remaining"]
    assert len(taken) >= 2
    assert len(remaining) >= 1
    assert any("package.json" in a for a in taken)
    assert any("migration_manifest" in a for a in taken)


def test_target_decision_documents_compatibility() -> None:
    """The decision must document plugin compatibility constraints."""
    decision = json.loads(TARGET_DECISION_PATH.read_text())
    constraints = decision["compatibility_constraints"]
    assert "starlight-links-validator" in constraints
    assert "starlight-versions" in constraints
    assert "@astrojs/starlight-docsearch" in constraints