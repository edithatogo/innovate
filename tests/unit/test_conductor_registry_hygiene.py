"""Regression tests for Conductor registry and filesystem hygiene."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CONDUCTOR_ROOT = Path("conductor")
TRACKS_MD = CONDUCTOR_ROOT / "tracks.md"
ACTIVE_ROOT = CONDUCTOR_ROOT / "tracks"
ARCHIVE_ROOT = CONDUCTOR_ROOT / "archive"
HYGIENE_SCRIPT = Path("scripts/conductor_registry_hygiene.py")


def _registry_entries() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    track_pattern = re.compile(r"- \[([ x~])\] \*\*Track: (.*?)\*\*")
    link_pattern = re.compile(r"\*Link: \[(.*?)\]\((.*?)\)\*")

    for line in TRACKS_MD.read_text(encoding="utf-8").splitlines():
        track_match = track_pattern.match(line)
        if track_match:
            current = {
                "status": track_match.group(1),
                "name": track_match.group(2),
                "link": "",
            }
            entries.append(current)
            continue

        link_match = link_pattern.search(line)
        if link_match and current is not None:
            current["link"] = link_match.group(2)

    return entries


def _folder_name(link: str) -> str:
    return Path(link.replace("./", "")).name


def test_active_track_folders_match_active_registry_entries() -> None:
    """Every active folder should have exactly one active registry entry."""
    entries = _registry_entries()
    active_folders = {path.name for path in ACTIVE_ROOT.iterdir() if path.is_dir()}
    registered_active = {
        _folder_name(entry["link"])
        for entry in entries
        if entry["status"] in {" ", "~"} and entry["link"].startswith("./tracks/")
    }

    assert active_folders == registered_active


def test_completed_registry_entries_point_to_existing_archive_folders() -> None:
    """Completed entries should resolve under conductor/archive."""
    for entry in _registry_entries():
        if entry["status"] != "x":
            continue

        assert entry["link"].startswith("./archive/"), entry
        assert (CONDUCTOR_ROOT / entry["link"].replace("./", "")).is_dir(), entry


def test_archive_folders_are_registered_or_explicitly_exempted() -> None:
    """Archived track evidence should be reachable from the registry."""
    entries = _registry_entries()
    registered_archive = {
        _folder_name(entry["link"])
        for entry in entries
        if entry["status"] == "x" and entry["link"].startswith("./archive/")
    }
    archive_folders = {path.name for path in ARCHIVE_ROOT.iterdir() if path.is_dir()}

    assert archive_folders == registered_archive


def test_status_tool_reports_registry_hygiene_diagnostics() -> None:
    """Status tooling should surface stale and orphaned folders directly."""
    assert HYGIENE_SCRIPT.is_file()

    result = subprocess.run(
        [sys.executable, str(HYGIENE_SCRIPT), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    for phrase in (
        "stale_active_directories",
        "orphan_archive_directories",
        "missing_archive_directories",
        "wrong_link_targets",
    ):
        assert phrase in result.stdout
