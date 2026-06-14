"""Tests for the Astro/Starlight route coverage report."""

from __future__ import annotations

import json
from pathlib import Path


def test_route_coverage_report_is_generated_from_inventories() -> None:
    """The coverage report should summarize implemented and planned routes."""
    report = json.loads(
        Path("docs/source/_static/astro_starlight/route_coverage.json").read_text()
    )

    assert report["generated_from"]["content_inventory"].endswith(
        "content_inventory.json"
    )
    assert report["counts"]["total"] == len(report["coverage_by_source_doc"])
    assert report["counts"]["implemented"] == report["counts"]["total"]
    assert report["counts"]["planned"] == 0

    implemented = {
        entry["source_doc"]
        for entry in report["coverage_by_source_doc"]
        if entry["status"] == "implemented"
    }
    planned = {
        entry["source_doc"]
        for entry in report["coverage_by_source_doc"]
        if entry["status"] == "planned"
    }

    assert implemented
    assert "docs/source/astro_starlight_migration.rst" in implemented
    assert not planned
    assert "docs/source/binding_publication_ci.rst" in implemented


def test_route_coverage_report_matches_astro_content_tree() -> None:
    """The coverage report should reflect actual files in the Astro scaffold."""
    report = json.loads(
        Path("docs/source/_static/astro_starlight/route_coverage.json").read_text()
    )

    implemented_routes = {
        entry["astro_route"]
        for entry in report["coverage_by_source_doc"]
        if entry["status"] == "implemented"
    }

    for route in (
        "/",
        "/api/python/",
        "/core/kernel/",
        "/core/arrow-interchange/",
        "/maintainers/publication/",
        "/user-guide/getting-started/",
        "/migration/redirects/",
        "/migration/validation/",
        "/migration/archive/",
        "/migration/references/",
    ):
        assert route in implemented_routes
