"""Tests for the Starlight dependency contract enforcement.

These tests validate that the documented Starlight baseline (from the target
decision artifact) is consistent with the actual package manifest, lockfile,
and migration evidence. Tests that assert the NEW target contract should FAIL
until the contract is enforced in Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path

TARGET_DECISION_PATH = Path("docs/source/_static/astro_starlight/starlight_target_decision.json")
PACKAGE_PATH = Path("docs/astro-site/package.json")
LOCKFILE_PATH = Path("docs/astro-site/pnpm-lock.yaml")
MANIFEST_PATH = Path("docs/source/_static/astro_starlight/migration_manifest.json")

EXPECTED_STARLIGHT_SPECIFIER = "^0.41.0"
EXPECTED_MANIFEST_STARLIGHT = "0.41.0"


def test_package_json_starlight_specifier_matches_target() -> None:
    """package.json @astrojs/starlight specifier must match the promoted target.

    EXPECTED TO FAIL until Phase 2 applies the target promotion.
    """
    package = json.loads(PACKAGE_PATH.read_text())
    specifier = package["dependencies"]["@astrojs/starlight"]
    assert specifier == EXPECTED_STARLIGHT_SPECIFIER, (
        f"package.json has @astrojs/starlight {specifier}, "
        f"expected {EXPECTED_STARLIGHT_SPECIFIER} per target decision"
    )


def test_migration_manifest_starlight_matches_target() -> None:
    """The migration manifest baseline must reflect the promoted Starlight version."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["baseline"]["starlight"] == EXPECTED_MANIFEST_STARLIGHT, (
        f"Manifest baseline starlight is {manifest['baseline']['starlight']}, "
        f"expected {EXPECTED_MANIFEST_STARLIGHT}"
    )


def test_package_json_lockfile_starlight_specifiers_agree() -> None:
    """The package.json and lockfile starlight specifiers must be consistent."""
    package = json.loads(PACKAGE_PATH.read_text())
    lines = _lockfile_starlight_lines(LOCKFILE_PATH.read_text())
    assert lines, "Could not find @astrojs/starlight entry in lockfile"

    package_spec = package["dependencies"]["@astrojs/starlight"]
    spec_line = next((l for l in lines if l.startswith("specifier:")), "")
    lockfile_spec = spec_line.replace("specifier:", "").strip().strip("'\"")

    assert package_spec == lockfile_spec, (
        f"package.json specifier ({package_spec}) does not match "
        f"lockfile specifier ({lockfile_spec})"
    )


def _lockfile_starlight_lines(lockfile_text: str) -> list[str]:
    """Extract lines from the @astrojs/starlight entry in the lockfile."""
    import re
    match = re.search(
        r"@astrojs/starlight':?\n((?:\s+[^\n]+\n)+)",
        lockfile_text,
    )
    if not match:
        return []
    return [line.strip() for line in match.group(1).strip().split("\n")]


def test_lockfile_starlight_resolved_version_is_41() -> None:
    """The lockfile must resolve to Starlight 0.41.x."""
    lines = _lockfile_starlight_lines(LOCKFILE_PATH.read_text())
    assert lines, "Could not find @astrojs/starlight entry in lockfile"

    spec_line = next((l for l in lines if l.startswith("specifier:")), "")
    ver_line = next((l for l in lines if l.startswith("version:")), "")

    assert "0.41." in ver_line, (
        f"Lockfile resolved to Starlight {ver_line}, expected 0.41.x"
    )


def test_target_decision_artifact_is_present() -> None:
    """The target decision artifact must exist for contract reference."""
    assert TARGET_DECISION_PATH.exists()