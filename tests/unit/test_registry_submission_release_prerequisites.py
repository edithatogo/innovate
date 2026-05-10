"""Tests for registry submission release prerequisites."""

from __future__ import annotations

import json
from pathlib import Path

PREREQUISITES_PATH = Path("docs/source/_static/registry_release_prerequisites.json")
VERSION = "0.5.0"


def load_prerequisites() -> dict[str, object]:
    return json.loads(PREREQUISITES_PATH.read_text(encoding="utf-8"))


def test_release_prerequisites_fixture_exists() -> None:
    """The release prerequisites should be captured in a machine-readable file."""
    assert PREREQUISITES_PATH.is_file()


def test_release_prerequisites_cover_all_package_targets() -> None:
    """Every package target should have a documented prerequisite summary."""
    prereqs = load_prerequisites()

    assert prereqs["schema_version"] == 1
    assert prereqs["version"] == VERSION
    assert set(prereqs["package_targets"]) == {
        "python_pypi",
        "typescript_npm",
        "rust_crates_io",
        "r_r_universe",
        "r_cran",
        "julia_general",
        "go_modules",
        "csharp_nuget",
    }


def test_release_prerequisites_record_gates_and_access_needs() -> None:
    """Each target should list gates and maintainer-access needs."""
    prereqs = load_prerequisites()

    for entry in prereqs["targets"]:
        assert entry["target_id"]
        assert entry["release_gate"]
        assert entry["required_access"] in {"maintainer", "registry_secret", "review"}
        assert entry["status"] in {"ready", "blocked", "deferred"}
        assert entry["notes"]

