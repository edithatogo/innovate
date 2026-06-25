"""Regression tests for Astro/Starlight cutover cleanup."""

from __future__ import annotations

import json
from pathlib import Path

CUTOVER_INVENTORY = Path("docs/source/_static/astro_starlight/cutover_surface_inventory.json")
STARLIGHT_VALIDATION = Path("docs/source/_static/astro_starlight/starlight_validation_evidence.json")
MIGRATION_MANIFEST = Path("docs/source/_static/astro_starlight/migration_manifest.json")


def _normalized_text(path: Path) -> str:
    return " ".join(path.read_text().split()).lower()


def test_product_and_tech_stack_name_starlight_as_active_docs_stack() -> None:
    """Product status and tech stack should agree that Starlight is active."""
    product = _normalized_text(Path("conductor/product.md"))
    tech_stack = _normalized_text(Path("conductor/tech-stack.md"))

    assert "documentation: astro/starlight" in product
    assert "site generator" in tech_stack
    assert "astro" in tech_stack
    assert "starlight" in tech_stack

    forbidden = (
        "documentation: sphinx",
        "sphinx site remains canonical",
        "sphinx remains canonical",
        "being migrated to starlight",
    )
    for phrase in forbidden:
        assert phrase not in product
        assert phrase not in tech_stack


def test_completed_starlight_migration_tracks_are_not_active_tracks() -> None:
    """Completed Starlight migration evidence belongs in the archive, not tracks."""
    active_tracks = {path.parent.name for path in Path("conductor/tracks").glob("*/metadata.json")}
    archived_tracks = {path.parent.name for path in Path("conductor/archive").glob("*/metadata.json")}

    stale_tracks = {"migrate_starlight", "starlight_migration_20260513"}
    assert stale_tracks.isdisjoint(active_tracks)
    assert stale_tracks <= archived_tracks


def test_migration_manifest_records_cutover_complete_not_parallel_run() -> None:
    """The manifest should not present another docs generator as canonical."""
    manifest = json.loads(MIGRATION_MANIFEST.read_text())

    assert manifest["migration_mode"] == "cutover-complete"
    assert manifest["active_docs_stack"] == "astro_starlight"
    assert manifest["legacy_docs_stack"] in {"removed", "none"}
    assert manifest["legacy_retention_policy"] in {"removed", "none"}
    assert manifest["route_stability_policy"] == "astro-native-routes-only"
    assert all("canonical" not in note.lower() for note in manifest["notes"])


def test_no_sphinx_tooling_or_active_docs_claims_remain() -> None:
    """The active repo surface should not reintroduce Sphinx tooling."""
    checked_paths = [
        Path("pyproject.toml"),
        Path("noxfile.py"),
        Path("docs/astro-site/README.md"),
        Path("docs/astro-site/src/content/docs/migration/index.md"),
        Path("docs/astro-site/src/content/docs/migration/redirects.md"),
        Path("docs/astro-site/src/content/docs/migration/validation.md"),
        Path("docs/astro-site/src/content/docs/core/arrow-interchange.md"),
        Path("docs/astro-site/src/content/docs/maintainers/publication.md"),
        Path("docs/astro-site/src/content/docs/maintainers/release-notes.md"),
    ]

    forbidden = (
        "python -m sphinx",
        "sphinx-build",
        "sphinx-rtd-theme",
        "sphinx-autodoc-typehints",
        "parallel-run",
        "canonical sphinx",
        "sphinx site",
    )

    for path in checked_paths:
        text = _normalized_text(path)
        for phrase in forbidden:
            assert phrase not in text, f"{path} still contains stale wording"

    removed_files = [
        Path("docs/conf.py"),
        Path("docs/Makefile"),
        Path("docs/make.bat"),
        Path("docs/deploy.sh"),
        Path("docs/source/conf.py"),
        Path("docs/source/make.bat"),
    ]
    assert not any(path.exists() for path in removed_files)


def test_cutover_inventory_has_no_remaining_stale_work_after_cleanup() -> None:
    """The cleanup inventory should become a zero-stale-work status artifact."""
    inventory = json.loads(CUTOVER_INVENTORY.read_text())

    assert inventory["active_docs_stack"] == "astro_starlight"
    assert inventory["legacy_surfaces"]["retention_policy"] in {"removed", "none"}
    assert inventory["stale_active_track_folders"] == []
    assert inventory["stale_cutover_language"] == []
    assert inventory["version_status"]["docsearch_package_present"] is True
    assert "blocked_or_deferred_items" not in inventory
    external_gates = {entry["item"]: entry for entry in inventory["external_gate_items"]}
    assert external_gates["DocSearch credentials"]["status"] == "external_credentials_required"
    resolved_items = {entry["item"]: entry for entry in inventory["resolved_items"]}
    assert resolved_items["starlight-versions active middleware"]["status"] == "enabled"


def test_starlight_validation_evidence_records_routes_and_build_status() -> None:
    """Route/link validation and the active Starlight build should both pass."""
    evidence = json.loads(STARLIGHT_VALIDATION.read_text())

    assert evidence["route_and_link_status"]["route_coverage"] == "passed"
    assert evidence["route_and_link_status"]["link_validation"] == "passed"
    assert evidence["route_and_link_status"]["broken_links"] == 0

    commands = {entry["command"]: entry for entry in evidence["commands"]}
    assert commands["pnpm install --frozen-lockfile"]["status"] == ("passed_after_workspace_build_approvals")
    assert commands["pnpm build && pnpm check"]["status"] == "passed"

    resolved = {entry["id"]: entry for entry in evidence["resolved_blockers"]}
    assert resolved["starlight_polyglot_missing_python_handler_bundle"]["status"] == ("resolved")
    assert resolved["starlight_versions_astro7_404_middleware"]["status"] == "resolved"
    assert "blockers" not in evidence
    external_gates = {entry["id"]: entry for entry in evidence["external_gates"]}
    assert external_gates["docsearch_credentials"]["status"] == "external_credentials_required"
    assert external_gates["astro_expressive_code_peer_range"]["status"] == "upstream_peer_metadata_lag"
    assert "starlight_versions_astro7_404_middleware" not in external_gates
