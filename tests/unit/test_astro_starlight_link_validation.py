"""Tests for the Astro/Starlight link validation report."""

from __future__ import annotations

import json
from pathlib import Path


def test_link_validation_report_confirms_sidebar_routes() -> None:
    """The report should confirm the sidebar routes are implemented."""
    report = json.loads(
        Path("docs/source/_static/astro_starlight/link_validation_report.json").read_text()
    )

    assert report["generated_from"]["route_coverage"].endswith("route_coverage.json")
    assert report["counts"]["sidebar_routes"] == 11
    assert report["counts"]["implemented_sidebar_routes"] == 11
    assert report["counts"]["broken_links"] == 0
    assert report["ready_for_route_stability"] is True

    sidebar_routes = {entry["route"] for entry in report["sidebar_route_validation"]}
    for route in (
        "/core/kernel/",
        "/core/arrow-interchange/",
        "/maintainers/publication/",
        "/migration/",
        "/migration/redirects/",
        "/migration/validation/",
        "/migration/archive/",
        "/migration/references/",
    ):
        assert route in sidebar_routes


def test_link_validation_report_covers_internal_route_links() -> None:
    """The report should verify the migration pages link to valid routes."""
    report = json.loads(
        Path("docs/source/_static/astro_starlight/link_validation_report.json").read_text()
    )

    checked_routes = set(report["checked_routes"])
    for route in (
        "/migration/redirects/",
        "/migration/validation/",
        "/migration/archive/",
        "/migration/references/",
    ):
        assert route in checked_routes

    for entry in report["link_checks"]:
        if entry["kind"] == "route":
            assert entry["implemented"] is True
