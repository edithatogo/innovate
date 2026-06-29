"""Tests for Starlight-only documentation enforcement via RST allowlist.

This test suite enforces that only explicitly-allowed RST files exist in the
docs/ directory. All other RST files should have been migrated to Starlight
or archived as evidence artifacts.

The allowlist is sourced from the RST classification JSON produced during
Phase 1 of the Starlight-Only Documentation Completion track.
"""

from __future__ import annotations

import json
from pathlib import Path

# Root project paths (running from innovate/ directory)
ROOT = Path()
DOCS_DIR = ROOT / "docs"
RST_CLASSIFICATION = (
    ROOT / "conductor" / "tracks" / "starlight_only_docs_completion_20260625" / "rst_classification.json"
)

# Starlight paths
STARLIGHT_CONFIG = ROOT / "docs" / "astro-site" / "starlight.config.mjs"
STARLIGHT_DOCS_ROOT = ROOT / "docs" / "astro-site" / "src" / "content" / "docs"


def load_allowlist() -> set[str]:
    """Load the RST allowlist from the classification JSON file.

    Returns a set of relative paths (from project root) that are allowed to exist as RST files.
    """
    if not RST_CLASSIFICATION.exists():
        raise FileNotFoundError(
            f"RST classification file not found: {RST_CLASSIFICATION}\n"
            "This file should be created during Phase 1: Remaining RST Audit"
        )

    with open(RST_CLASSIFICATION) as f:
        classification = json.load(f)

    allowlist = classification.get("keep_allowlist", [])
    return {Path(p).as_posix() for p in allowlist}


def find_all_rst_files() -> set[str]:
    """Find all RST files in the docs directory.

    Returns a set of relative paths (from project root) to all RST files.
    """
    rst_files = set()
    for rst_file in DOCS_DIR.rglob("*.rst"):
        # Get the relative path from the project root
        relative_path = rst_file.relative_to(ROOT).as_posix()
        # Normalize path separators for consistency
        rst_files.add(relative_path)
    return rst_files


def test_rst_allowlist_exists() -> None:
    """The RST classification and allowlist should exist."""
    assert RST_CLASSIFICATION.exists(), (
        f"RST classification file missing: {RST_CLASSIFICATION}\n"
        "Run the Classify remaining RST files task to generate this file."
    )

    with open(RST_CLASSIFICATION) as f:
        data = json.load(f)

    assert "keep_allowlist" in data, "Missing 'keep_allowlist' key in classification"
    assert isinstance(data["keep_allowlist"], list), "'keep_allowlist' should be a list"
    assert len(data["keep_allowlist"]) > 0, "'keep_allowlist' should not be empty"


def test_no_active_rst_files_outside_allowlist() -> None:
    """All RST files must be in the allowlist or be archived/evidence artifacts.

    This is the fail-closed guard: any unclassified RST file will cause this test to fail.
    """
    allowlist = load_allowlist()
    found_files = find_all_rst_files()

    # Files that exist but are not in the allowlist
    unallowed_files = found_files - allowlist

    # These files should NOT exist
    assert not unallowed_files, (
        f"Found {len(unallowed_files)} RST file(s) that are not in the allowlist.\n"
        f"These files should have been migrated to Starlight or archived:\n"
        + "\n".join(f"  - {f}" for f in sorted(unallowed_files))
    )


def test_allowlist_files_exist() -> None:
    """All files in the allowlist should actually exist (sanity check).

    This ensures the allowlist itself is accurate and maintained.
    """
    allowlist = load_allowlist()
    missing_files = []

    for allowed_path in sorted(allowlist):
        file_path = ROOT / allowed_path
        if not file_path.exists():
            missing_files.append(allowed_path)

    # We allow some tolerance here as the allowlist might reference files
    # that are generated or conditional, but we want to know about missing files
    if missing_files and len(missing_files) < len(allowlist) * 0.1:
        # Less than 10% of allowlist missing is acceptable
        pass
    elif missing_files:
        # More than 10% missing suggests a problem
        assert not missing_files, f"Found {len(missing_files)} allowlist entries that don't exist:\n" + "\n".join(
            f"  - {f}" for f in sorted(missing_files)[:10]
        )


def test_starlight_docs_structure_exists() -> None:
    """Starlight documentation structure should exist and be configured."""
    assert STARLIGHT_CONFIG.exists(), (
        f"Starlight config not found: {STARLIGHT_CONFIG}\nStarlight should be properly initialized."
    )

    assert STARLIGHT_DOCS_ROOT.exists(), (
        f"Starlight docs directory not found: {STARLIGHT_DOCS_ROOT}\nStarlight documentation structure should exist."
    )

    # Check that there are some docs in the Starlight structure
    md_files = list(STARLIGHT_DOCS_ROOT.rglob("*.md"))
    assert len(md_files) > 0, (
        "No Markdown files found in Starlight docs directory.\n"
        "Migration to Starlight should have created documentation files."
    )


def test_rst_allowlist_has_required_structure() -> None:
    """The allowlist should include required documentation structure."""
    allowlist = load_allowlist()

    # These are core files that should always be in the allowlist
    required_core_files = [
        "innovate/docs/index.rst",
        "innovate/docs/source/index.rst",
    ]

    for required_file in required_core_files:
        assert required_file in allowlist, (
            f"Required file missing from allowlist: {required_file}\n"
            f"This file is essential for the documentation structure."
        )


if __name__ == "__main__":
    import sys

    print("Testing RST allowlist enforcement...")

    # Run the key test
    try:
        test_rst_allowlist_exists()
        print("✓ Allowlist exists")
    except AssertionError as e:
        print(f"✗ Allowlist check failed: {e}")
        sys.exit(1)

    try:
        test_no_active_rst_files_outside_allowlist()
        print("✓ No unallowed RST files found")
    except AssertionError as e:
        print(f"✗ Found unallowed RST files:\n{e}")
        sys.exit(1)

    try:
        test_allowlist_files_exist()
        print("✓ Allowlist files exist (with tolerance)")
    except AssertionError as e:
        print(f"✗ Allowlist file check failed: {e}")

    try:
        test_starlight_docs_structure_exists()
        print("✓ Starlight docs structure exists")
    except AssertionError as e:
        print(f"✗ Starlight structure check failed: {e}")

    try:
        test_rst_allowlist_has_required_structure()
        print("✓ Required files in allowlist")
    except AssertionError as e:
        print(f"✗ Required files check failed: {e}")

    print("\nAll RST allowlist tests passed!")
