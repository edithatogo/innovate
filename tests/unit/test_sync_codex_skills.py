"""Tests for the repo-managed Codex skill sync utility."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_sync_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "sync_codex_skills.py"
    spec = importlib.util.spec_from_file_location("sync_codex_skills", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load sync_codex_skills module")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_discover_skill_directories_only_returns_skill_dirs(tmp_path: Path):
    module = _load_sync_module()
    skills_root = tmp_path / "skills"
    (skills_root / "valid-skill").mkdir(parents=True)
    (skills_root / "valid-skill" / "SKILL.md").write_text("name: valid\n", encoding="utf-8")
    (skills_root / "not-a-skill").mkdir(parents=True)
    (skills_root / "README.md").write_text("repo docs\n", encoding="utf-8")

    discovered = module.discover_skill_directories(skills_root)

    assert [skill.name for skill in discovered] == ["valid-skill"]


def test_resolve_codex_home_prefers_explicit_and_env(tmp_path: Path, monkeypatch):
    module = _load_sync_module()
    explicit = tmp_path / "explicit-home"
    env_home = tmp_path / "env-home"

    monkeypatch.setenv("CODEX_HOME", str(env_home))
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path / "default-home")

    assert module.resolve_codex_home(str(explicit)) == explicit.resolve()
    assert module.resolve_codex_home() == env_home.resolve()


def test_sync_skills_copies_nested_files(tmp_path: Path):
    module = _load_sync_module()
    source_root = tmp_path / "source-skills"
    destination_root = tmp_path / "codex-home" / "skills"

    skill_dir = source_root / "conductor-review"
    asset_dir = skill_dir / "references"
    asset_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("name: conductor-review\n", encoding="utf-8")
    (asset_dir / "notes.txt").write_text("reference material\n", encoding="utf-8")

    installed = module.sync_skills(source_root, destination_root)

    assert installed == ["conductor-review"]
    assert (destination_root / "conductor-review" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "name: conductor-review\n"
    assert (destination_root / "conductor-review" / "references" / "notes.txt").read_text(
        encoding="utf-8"
    ) == "reference material\n"
