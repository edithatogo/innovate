"""Tests for the Astro/Starlight migration scaffold and inventories."""

from __future__ import annotations

import json
from pathlib import Path


def test_astro_starlight_package_manifest_records_pinned_baseline() -> None:
    """The active site manifest should record the documented Starlight baseline."""
    package = json.loads(Path("docs/astro-site/package.json").read_text())

    assert package["name"] == "innovate-docs"
    assert package["private"] is True
    assert package["scripts"]["build"] == "astro build"
    assert package["scripts"]["dev"] == "astro dev"
    assert package["scripts"]["check"] == "astro check"
    assert Path("docs/astro-site/pnpm-lock.yaml").exists()

    dependencies = package["dependencies"]
    assert dependencies["astro"] == "^6.0.0"
    assert dependencies["@astrojs/starlight"] == "^0.39.0"
    assert dependencies["starlight-versions"] == "^0.5.4"
    assert dependencies["starlight-links-validator"] == "^0.24.0"
    assert dependencies["starlight-polyglot"].startswith("file:")


def test_astro_and_starlight_config_files_record_the_scaffold() -> None:
    """The scaffold config files should name the chosen integration surface."""
    astro_config = Path("docs/astro-site/astro.config.mjs").read_text()
    starlight_config = Path("docs/astro-site/starlight.config.mjs").read_text()

    for phrase in (
        "@astrojs/starlight",
        "Innovate",
        "starlight-polyglot",
    ):
        assert phrase in astro_config or phrase in starlight_config

    content_config = Path("docs/astro-site/src/content.config.ts").read_text()
    assert "docsLoader()" in content_config
    assert "docsSchema()" in content_config
    assert "@astrojs/starlight/loaders" in content_config
    assert "@astrojs/starlight/schema" in content_config

    for phrase in (
        "starlightLinksValidator",
        "starlight-versions",
        "starlight-links-validator",
        "@astrojs/starlight-docsearch",
        "Kernel",
        "Arrow Interchange",
        "Diagnostics",
        "Rust Core",
        "Release Notes",
        "Bindings",
        "Publication",
        "Migration",
        "Validation",
    ):
        assert phrase in starlight_config


def test_astro_starlight_migration_manifest_records_cutover_decisions() -> None:
    """The migration manifest should record the cutover policy explicitly."""
    manifest = json.loads(Path("docs/source/_static/astro_starlight/migration_manifest.json").read_text())

    assert manifest["migration_mode"] == "cutover-complete"
    assert manifest["active_docs_stack"] == "astro_starlight"
    assert manifest["legacy_docs_stack"] == "sphinx"
    assert manifest["legacy_retention_policy"] == "archival_and_redirect_reference_only"
    assert manifest["search_provider"] == "algolia-docsearch"
    assert manifest["sitemap_provider"] == "@astrojs/sitemap"
    assert manifest["baseline"]["astro"] == "^6.0.0"
    assert manifest["baseline"]["starlight"] == "0.38.4"
    assert manifest["baseline"]["starlight_versions"] == "0.5.4"
    assert manifest["baseline"]["starlight_links_validator"] == "0.24.0"
    assert manifest["baseline"]["starlight_docsearch"] == "0.6.1"
    assert manifest["scaffold_root"] == "docs/astro-site"
    assert manifest["route_stability_policy"] == "compatibility-aliases-for-legacy-sphinx-urls"


def test_astro_starlight_inventories_stay_synchronized() -> None:
    """Content and redirect inventories should describe the same pages."""
    content_inventory = json.loads(Path("docs/source/_static/astro_starlight/content_inventory.json").read_text())
    redirect_inventory = json.loads(Path("docs/source/_static/astro_starlight/redirect_inventory.json").read_text())

    assert len(content_inventory) == len(redirect_inventory)
    assert {entry["source_doc"] for entry in content_inventory} == {entry["source_doc"] for entry in redirect_inventory}
    assert {entry["astro_route"] for entry in content_inventory} == {
        entry["astro_route"] for entry in redirect_inventory
    }


def test_astro_starlight_docs_page_lists_the_scaffold_artifacts() -> None:
    """The Sphinx-facing migration page should name the scaffold artifacts."""
    page = Path("docs/source/astro_starlight_migration.rst").read_text()

    for phrase in (
        "Astro/Starlight documentation site migration",
        "cutover-complete",
        "Algolia DocSearch",
        "content inventory",
        "redirect inventory",
        "route_coverage.json",
        "cutover_verification.json",
        "link_validation_report.json",
        "generate_route_coverage.py",
        "generate_cutover_verification.py",
        "generate_link_validation.py",
        "docs/astro-site/package.json",
        "docs/source/_static/astro_starlight/migration_manifest.json",
    ):
        assert phrase in page


def test_astro_starlight_navigation_includes_the_migration_page() -> None:
    """The Sphinx site should surface the migration page in navigation."""
    index = Path("docs/source/index.rst").read_text()

    assert "astro_starlight_migration" in index


def test_astro_starlight_core_pages_have_migrated_content() -> None:
    """The first migrated Astro pages should carry actual documentation copy."""
    kernel = Path("docs/astro-site/src/content/docs/core/kernel.md").read_text()
    arrow = Path("docs/astro-site/src/content/docs/core/arrow-interchange.md").read_text()
    diagnostics = Path("docs/astro-site/src/content/docs/core/diagnostics-contract.md").read_text()
    rust_core = Path("docs/astro-site/src/content/docs/operations/rust-core.md").read_text()
    roadmap = Path("docs/astro-site/src/content/docs/operations/roadmap.md").read_text()

    for phrase in (
        "stable functional kernel",
        "Arrow remains the stable interchange boundary",
        "fitted-state reporting fields",
        "Rust owns the promoted native slices",
        "Python remains the reference ergonomic surface",
    ):
        assert phrase in kernel or phrase in arrow or phrase in diagnostics or phrase in rust_core or phrase in roadmap


def test_astro_starlight_bindings_and_publication_pages_have_content() -> None:
    """Bindings and publication landing pages should describe the migration surface."""
    bindings = Path("docs/astro-site/src/content/docs/bindings/index.md").read_text()
    publication = Path("docs/astro-site/src/content/docs/maintainers/publication.md").read_text()
    release_notes = Path("docs/astro-site/src/content/docs/maintainers/release-notes.md").read_text()

    for phrase in (
        "Python as the canonical reference surface",
        "Rust as the native runtime and binding target",
        "PyPI/TestPyPI",
        "npm",
        "crates.io",
        "R-universe",
        "CRAN",
        "Julia General",
        "Go modules",
        "NuGet",
    ):
        assert phrase in bindings or phrase in publication or phrase in release_notes


def test_astro_starlight_maintainers_area_describes_release_notes() -> None:
    """The maintainers area should surface the release-notes policy page."""
    maintainers = Path("docs/astro-site/src/content/docs/maintainers/index.md").read_text()
    release_notes = Path("docs/astro-site/src/content/docs/maintainers/release-notes.md").read_text()

    assert "Release Notes" in maintainers
    for phrase in (
        "Release Please",
        "Release Drafter",
        "Commitizen",
        "CHANGELOG.md",
    ):
        assert phrase in release_notes


def test_astro_starlight_redirect_route_map_describes_cutover() -> None:
    """The migration area should expose a route map for redirect coverage."""
    migration = Path("docs/astro-site/src/content/docs/migration/index.md").read_text()
    redirects = Path("docs/astro-site/src/content/docs/migration/redirects.md").read_text()
    validation = Path("docs/astro-site/src/content/docs/migration/validation.md").read_text()

    assert "redirect inventory" in migration
    assert "route-stability" in validation
    for phrase in (
        "docs/source/index.rst",
        "/core/kernel/",
        "/maintainers/publication/",
        "/maintainers/release-notes/",
        "/operations/rust-core/",
        "Legacy Sphinx URLs remain reachable",
    ):
        assert phrase in redirects

    for phrase in (
        "/migration/redirects/",
        "/migration/archive/",
        "/migration/references/",
    ):
        assert phrase in migration or phrase in validation


def test_astro_starlight_archive_and_reference_pages_describe_provenance() -> None:
    """The archive and references pages should guide readers to provenance."""
    archive = Path("docs/astro-site/src/content/docs/migration/archive.md").read_text()
    references = Path("docs/astro-site/src/content/docs/migration/references.md").read_text()
    migration = Path("docs/astro-site/src/content/docs/migration/index.md").read_text()

    for phrase in (
        "completed Conductor tracks",
        "Rust core ownership closure tracks",
        "Registry submission receipts",
        "HPC readiness artifacts",
        "Migration References",
        "route-stability",
        "polyglot_repo_architecture",
        "rust_core_roadmap",
        "registry_submission_execution_20260511",
    ):
        assert phrase in archive or phrase in references or phrase in migration
