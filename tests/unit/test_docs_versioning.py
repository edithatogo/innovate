"""Documentation versioning checks."""

from __future__ import annotations

from importlib.metadata import version as package_version
from pathlib import Path


def test_sphinx_conf_uses_package_version_metadata() -> None:
    """Sphinx release/version should not drift from package metadata."""
    namespace: dict[str, object] = {}
    conf_path = Path("docs/source/conf.py")

    exec(conf_path.read_text(encoding="utf-8"), {"__file__": str(conf_path)}, namespace)

    expected_version = package_version("innovate")
    assert namespace["release"] == expected_version
    assert namespace["version"] == expected_version
    assert 'release = "1.0.0"' not in conf_path.read_text(encoding="utf-8")
    assert 'version = "1.0.0"' not in conf_path.read_text(encoding="utf-8")
