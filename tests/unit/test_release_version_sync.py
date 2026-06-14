"""Tests for the canonical release-version sync guard."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


def _copy_tree(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    for rel in (
        "pyproject.toml",
        "scripts/sync_versions.py",
        "bindings/typescript/package.json",
        "bindings/julia/Project.toml",
        "bindings/rust/Cargo.toml",
        "bindings/r/DESCRIPTION",
        "bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj",
    ):
        src = ROOT / rel
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    return repo


def test_version_sync_check_passes_on_aligned_manifests(tmp_path: Path) -> None:
    """The checker should pass when the manifests match the canonical version."""
    repo = _copy_tree(tmp_path)
    result = subprocess.run(
        ["python3", str(repo / "scripts" / "sync_versions.py"), "--check", "--root", str(repo)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Version manifests are aligned" in result.stdout


def test_version_sync_check_fails_when_manifest_drift_exists(tmp_path: Path) -> None:
    """The checker should fail when a package manifest drifts."""
    repo = _copy_tree(tmp_path)
    package_json = repo / "bindings/typescript/package.json"
    data = json.loads(package_json.read_text())
    data["version"] = "9.9.9"
    package_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    result = subprocess.run(
        ["python3", str(repo / "scripts" / "sync_versions.py"), "--check", "--root", str(repo)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Version drift detected" in result.stdout
    assert "bindings/typescript/package.json" in result.stdout


def test_version_sync_write_updates_supported_manifests(tmp_path: Path) -> None:
    """The writer should bring supported manifests back to the canonical version."""
    repo = _copy_tree(tmp_path)
    pyproject = repo / "pyproject.toml"
    text = pyproject.read_text()
    pyproject.write_text(text.replace('version = "0.5.0"', 'version = "0.6.0"'), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(repo / "scripts" / "sync_versions.py"), "--write", "--root", str(repo)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "bindings/typescript/package.json" in result.stdout
    assert json.loads((repo / "bindings/typescript/package.json").read_text())["version"] == "0.6.0"
    assert tomllib.loads((repo / "bindings/julia/Project.toml").read_text())["version"] == "0.6.0"
    assert tomllib.loads((repo / "bindings/rust/Cargo.toml").read_text())["package"]["version"] == "0.6.0"
    assert "Version: 0.6.0" in (repo / "bindings/r/DESCRIPTION").read_text()
    assert "<Version>0.6.0</Version>" in (repo / "bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj").read_text()
