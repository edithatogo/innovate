"""Check Conductor track registry and filesystem hygiene."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONDUCTOR_ROOT = ROOT / "conductor"
TRACKS_MD = CONDUCTOR_ROOT / "tracks.md"
ACTIVE_ROOT = CONDUCTOR_ROOT / "tracks"
ARCHIVE_ROOT = CONDUCTOR_ROOT / "archive"


def _folder_name(link: str) -> str:
    return Path(link.replace("./", "")).name


def parse_registry() -> list[dict[str, str]]:
    """Parse track status and links from conductor/tracks.md."""
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


def build_report() -> dict[str, Any]:
    """Build a machine-readable registry/filesystem drift report."""
    entries = parse_registry()
    active_dirs = {path.name for path in ACTIVE_ROOT.iterdir() if path.is_dir()}
    archive_dirs = {path.name for path in ARCHIVE_ROOT.iterdir() if path.is_dir()}
    registered_active: set[str] = set()
    registered_archive: set[str] = set()
    missing_links: list[dict[str, str]] = []
    wrong_link_targets: list[dict[str, str]] = []

    for entry in entries:
        link = entry["link"]
        if not link:
            missing_links.append({**entry, "reason": "missing_link"})
            continue

        normalized = link.replace("./", "")
        target = CONDUCTOR_ROOT / normalized
        if not target.exists():
            missing_links.append({**entry, "reason": "target_missing"})

        if entry["status"] in {" ", "~"}:
            registered_active.add(_folder_name(link))
            if not normalized.startswith("tracks/"):
                wrong_link_targets.append({**entry, "reason": "active_not_tracks"})
        elif entry["status"] == "x":
            registered_archive.add(_folder_name(link))
            if not normalized.startswith("archive/"):
                wrong_link_targets.append({**entry, "reason": "completed_not_archive"})

    stale_active = sorted(active_dirs - registered_active)
    missing_active = sorted(registered_active - active_dirs)
    orphan_archive = sorted(archive_dirs - registered_archive)
    missing_archive = sorted(registered_archive - archive_dirs)

    return {
        "schema_version": 1,
        "registry": "conductor/tracks.md",
        "active_root": "conductor/tracks",
        "archive_root": "conductor/archive",
        "summary": {
            "registry_entries": len(entries),
            "active_directories": len(active_dirs),
            "registered_active_entries": len(registered_active),
            "stale_active_directories": len(stale_active),
            "missing_active_directories": len(missing_active),
            "orphan_archive_directories": len(orphan_archive),
            "missing_archive_directories": len(missing_archive),
            "missing_links": len(missing_links),
            "wrong_link_targets": len(wrong_link_targets),
        },
        "registered_active_directories": sorted(registered_active),
        "stale_active_directories": stale_active,
        "missing_active_directories": missing_active,
        "orphan_archive_directories": orphan_archive,
        "missing_archive_directories": missing_archive,
        "missing_links": missing_links,
        "wrong_link_targets": wrong_link_targets,
    }


def has_drift(report: dict[str, Any]) -> bool:
    """Return True when any hygiene diagnostics contain actionable drift."""
    return any(
        report[key]
        for key in (
            "stale_active_directories",
            "missing_active_directories",
            "orphan_archive_directories",
            "missing_archive_directories",
            "missing_links",
            "wrong_link_targets",
        )
    )


def render_text(report: dict[str, Any]) -> str:
    """Render a concise human-readable status report."""
    lines = ["Conductor registry hygiene"]
    for key, value in report["summary"].items():
        lines.append(f"- {key}: {value}")

    for key in (
        "stale_active_directories",
        "missing_active_directories",
        "orphan_archive_directories",
        "missing_archive_directories",
        "missing_links",
        "wrong_link_targets",
    ):
        value = report[key]
        lines.append(f"- {key}: {value or '[]'}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    args = parser.parse_args(argv)

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))

    return 1 if has_drift(report) else 0


if __name__ == "__main__":
    sys.exit(main())
