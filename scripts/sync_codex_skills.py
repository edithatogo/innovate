#!/usr/bin/env python3
"""Sync repo-managed Codex skills into the active Codex home."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillDirectory:
    """A repo-managed Codex skill directory."""

    name: str
    path: Path


def resolve_codex_home(explicit_target: str | None = None) -> Path:
    """Resolve the Codex home directory from explicit input, env, or default."""
    if explicit_target:
        return Path(explicit_target).expanduser().resolve()

    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser().resolve()

    return (Path.home() / ".codex").resolve()


def discover_skill_directories(skills_root: Path) -> list[SkillDirectory]:
    """Return skill directories that contain a SKILL.md file."""
    if not skills_root.exists():
        raise FileNotFoundError(f"Skill source directory does not exist: {skills_root}")

    skills: list[SkillDirectory] = []
    for child in sorted(skills_root.iterdir()):
        if child.is_dir() and (child / "SKILL.md").is_file():
            skills.append(SkillDirectory(name=child.name, path=child))
    return skills


def copy_skill_tree(source: Path, destination: Path) -> None:
    """Overlay-copy a skill directory, preserving nested files."""
    destination.mkdir(parents=True, exist_ok=True)

    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def sync_skills(source_root: Path, destination_root: Path) -> list[str]:
    """Copy all repo-managed skills to the destination skills directory."""
    installed: list[str] = []
    for skill in discover_skill_directories(source_root):
        copy_skill_tree(skill.path, destination_root / skill.name)
        installed.append(skill.name)
    return installed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Sync repo-managed Codex skills into the active Codex home.",
    )
    parser.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parents[1] / ".codex" / "skills"),
        help="Source directory containing repo-managed skills.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Codex home directory to install into. Defaults to $CODEX_HOME or ~/.codex.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return parser


def main() -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    codex_home = resolve_codex_home(args.target)
    destination_root = codex_home / "skills"

    skills = discover_skill_directories(source_root)
    if not skills:
        print(f"No skills found in {source_root}")
        return 1

    if args.dry_run:
        print(f"Would sync {len(skills)} skill(s) from {source_root} to {destination_root}:")
        for skill in skills:
            print(f"- {skill.name}")
        return 0

    installed = sync_skills(source_root, destination_root)
    print(f"Synced {len(installed)} skill(s) to {destination_root}:")
    for skill_name in installed:
        print(f"- {skill_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
