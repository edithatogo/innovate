"""Tests for the Astro/Starlight cutover verification report."""

from __future__ import annotations

import json
from pathlib import Path


def test_cutover_verification_report_confirms_inventory_alignment() -> None:
    """The content and redirect inventories should be fully aligned."""
    report = json.loads(
        Path(
            "docs/source/_static/astro_starlight/cutover_verification.json"
        ).read_text()
    )

    assert report["generated_from"]["content_inventory"].endswith(
        "content_inventory.json"
    )
    assert report["generated_from"]["redirect_inventory"].endswith(
        "redirect_inventory.json"
    )
    assert report["counts"]["content_entries"] == report["counts"]["redirect_entries"]
    assert report["counts"]["content_only"] == 0
    assert report["counts"]["redirect_only"] == 0
    assert report["counts"]["matched_entries"] == report["counts"]["content_entries"]
    assert report["ready_for_cutover"] is True

    for entry in report["matched_entries"]:
        assert entry["routes_match"] is True
        assert entry["redirect_type_ok"] is True
        assert entry["redirect_type"] == "temporary-forward"


def test_cutover_verification_report_covers_expected_routes() -> None:
    """The cutover verification report should cover the canonical routes."""
    report = json.loads(
        Path(
            "docs/source/_static/astro_starlight/cutover_verification.json"
        ).read_text()
    )

    routes = {entry["astro_route"] for entry in report["matched_entries"]}
    assert "/" in routes
    assert "/core/kernel/" in routes
    assert "/maintainers/publication/" in routes
    assert "/operations/rust-core/" in routes
    assert "/architecture/adr/" in routes
