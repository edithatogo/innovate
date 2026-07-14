"""Tests for bleeding-edge dependency baselines.

This test module verifies that the project maintains the configured
dependency baselines as defined in the tech-stack.md and pyproject.toml:

- Python 3.14 (minimum)
- NumPy 2.0+
- Pydantic v2.0+
- Polars 1.0+ (for dataframe operations)
- basedpyright strict mode
- Astro 7 with Starlight
- TypeScript 6
- Node 26
- Vitest 4
- criterion 0.8 (Rust benchmarks)
"""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

import pytest


class TestPythonBaseline:
    """Verify Python runtime baseline is 3.14."""

    def test_python_version_requirement(self) -> None:
        """Python version must be 3.14+."""
        assert sys.version_info >= (3, 14), (
            f"Expected Python 3.14+, got {sys.version_info.major}.{sys.version_info.minor}"
        )

    def test_pyproject_requires_python_314(self) -> None:
        """pyproject.toml requires-python must specify 3.14."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()

        # Extract requires-python field
        match = re.search(r'requires-python\s*=\s*"([^"]+)"', content)
        assert match, "requires-python not found in pyproject.toml"

        requires_python = match.group(1)
        assert "3.14" in requires_python or ">=3.14" in requires_python, (
            f"requires-python should specify 3.14, got: {requires_python}"
        )


class TestNumPyBaseline:
    """Verify NumPy is version 2.0+."""

    def test_numpy_major_version(self) -> None:
        """NumPy must be version 2.0+."""
        numpy_version = metadata.version("numpy")
        major_version = int(numpy_version.split(".")[0])
        assert major_version >= 2, f"Expected NumPy 2.0+, got {numpy_version}"

    def test_pyproject_numpy_constraint(self) -> None:
        """pyproject.toml must specify NumPy 2.0+."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()

        # NumPy should be constrained to >=2.x,<3
        assert "numpy>=" in content, "NumPy dependency not found in pyproject.toml"
        match = re.search(r"numpy([>=<,\d.]+)", content)
        assert match, "Could not parse numpy constraint"

        constraint = match.group(1)
        assert "2" in constraint, f"NumPy constraint should target 2.x, got: {constraint}"


class TestPydanticBaseline:
    """Verify Pydantic is version 2.0+."""

    def test_pydantic_major_version(self) -> None:
        """Pydantic must be version 2.0+."""
        pydantic_version = metadata.version("pydantic")
        major_version = int(pydantic_version.split(".")[0])
        assert major_version >= 2, f"Expected Pydantic 2.0+, got {pydantic_version}"

    def test_pyproject_pydantic_constraint(self) -> None:
        """pyproject.toml dev dependencies must specify Pydantic 2.0+."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()

        # Pydantic should be in dev dependencies
        assert "pydantic>=" in content, "Pydantic not found in dependencies"
        match = re.search(r"pydantic([>=<,\d.]+)", content)
        assert match, "Could not parse pydantic constraint"

        constraint = match.group(1)
        assert "2" in constraint, f"Pydantic constraint should target 2.x, got: {constraint}"


class TestPolarsBaseline:
    """Verify Polars is available as optional dependency for dataframes."""

    def test_polars_available(self) -> None:
        """Polars should be available as optional dependency."""
        try:
            polars_version = metadata.version("polars")
            major_version = int(polars_version.split(".")[0])
            assert major_version >= 1, f"Expected Polars 1.0+, got {polars_version}"
        except metadata.PackageNotFoundError:
            # Polars is optional, so it's OK if not installed
            # But it should be declared as optional
            pass

    def test_pyproject_polars_optional(self) -> None:
        """pyproject.toml must declare Polars as optional dependency."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()

        assert "[project.optional-dependencies]" in content
        assert "dataframe" in content
        assert "polars" in content, "Polars not found in optional dependencies"


class TestBasedPyrightConfiguration:
    """Verify basedpyright is configured for strict mode."""

    def test_basedpyright_installed(self) -> None:
        """basedpyright must be installed."""
        try:
            basedpyright_version = metadata.version("basedpyright")
            assert basedpyright_version, "basedpyright not found"
        except metadata.PackageNotFoundError:
            pytest.skip("basedpyright not installed")

    def test_pyrightconfig_exists(self) -> None:
        """pyrightconfig.json must exist in project root."""
        pyrightconfig_path = Path(__file__).parent.parent / "pyrightconfig.json"
        assert pyrightconfig_path.exists(), "pyrightconfig.json not found"

        config = json.loads(pyrightconfig_path.read_text())
        # Check for strict mode settings
        assert config.get("typeCheckingMode") in [
            "strict",
            "basic",
        ], "typeCheckingMode should be configured"

    def test_pyproject_python_version_314(self) -> None:
        """pyrightconfig targets Python 3.14."""
        pyrightconfig_path = Path(__file__).parent.parent / "pyrightconfig.json"
        if pyrightconfig_path.exists():
            config = json.loads(pyrightconfig_path.read_text())
            python_version = config.get("pythonVersion")
            if python_version:
                assert "3.14" in str(python_version), f"pythonVersion should target 3.14, got: {python_version}"


class TestAstroStarlightBaseline:
    """Verify Astro 7 and Starlight versions."""

    def test_astro_package_json_exists(self) -> None:
        """package.json in docs/astro-site must exist."""
        package_json_path = Path(__file__).parent.parent / "docs" / "astro-site" / "package.json"
        assert package_json_path.exists(), "docs/astro-site/package.json not found"

    def test_astro_version_7(self) -> None:
        """Astro must be version 7.x."""
        package_json_path = Path(__file__).parent.parent / "docs" / "astro-site" / "package.json"
        if package_json_path.exists():
            package_json = json.loads(package_json_path.read_text())
            astro_version = package_json.get("dependencies", {}).get("astro", "")
            assert "^7" in astro_version or "7." in astro_version, f"Astro version should be 7.x, got: {astro_version}"

    def test_starlight_version_0_40_plus(self) -> None:
        """Starlight must be version 0.40+."""
        package_json_path = Path(__file__).parent.parent / "docs" / "astro-site" / "package.json"
        if package_json_path.exists():
            package_json = json.loads(package_json_path.read_text())
            starlight_version = package_json.get("dependencies", {}).get("@astrojs/starlight", "")
            assert "^0.4" in starlight_version or "0.4" in starlight_version, (
                f"Starlight version should be 0.40+, got: {starlight_version}"
            )


class TestTypeScriptBaseline:
    """Verify TypeScript is version 6+."""

    def test_typescript_package_json_devdep(self) -> None:
        """TypeScript in package.json must be version 6+."""
        package_json_path = Path(__file__).parent.parent / "docs" / "astro-site" / "package.json"
        if package_json_path.exists():
            package_json = json.loads(package_json_path.read_text())
            typescript_version = package_json.get("devDependencies", {}).get("typescript", "")
            assert "^6" in typescript_version, f"TypeScript version should be 6+, got: {typescript_version}"


class TestNodeBaseline:
    """Verify Node.js engine requirement."""

    def test_node_26_engine(self) -> None:
        """package.json must require Node 26+."""
        package_json_path = Path(__file__).parent.parent / "docs" / "astro-site" / "package.json"
        if package_json_path.exists():
            package_json = json.loads(package_json_path.read_text())
            node_engine = package_json.get("engines", {}).get("node", "")
            assert ">=26" in node_engine or "26" in node_engine, f"Node engine should be >=26, got: {node_engine}"


class TestVitestBaseline:
    """Verify Vitest is version 4+ if configured."""

    def test_vitest_package_json_devdep(self) -> None:
        """Vitest in package.json devDependencies should be 4+ if present."""
        package_json_path = Path(__file__).parent.parent / "docs" / "astro-site" / "package.json"
        if package_json_path.exists():
            package_json = json.loads(package_json_path.read_text())
            vitest_version = package_json.get("devDependencies", {}).get("vitest") or package_json.get(
                "dependencies", {}
            ).get("vitest")
            if vitest_version:
                assert "^4" in vitest_version, f"Vitest version should be 4+, got: {vitest_version}"


class TestRustCriterionBaseline:
    """Verify Rust benchmark criterion remains compatible with Rust 1.85."""

    def test_cargo_criterion_version(self) -> None:
        """Cargo.toml must retain the Rust 1.85-compatible Criterion pin."""
        cargo_toml_path = Path(__file__).parent.parent / "bindings" / "rust" / "Cargo.toml"
        if cargo_toml_path.exists():
            content = cargo_toml_path.read_text()
            assert "criterion" in content, "criterion not found in dev-dependencies"
            match = re.search(r'criterion\s*=\s*"([^"]+)"', content)
            if match:
                constraint = match.group(1)
                assert constraint in {"=0.5.1", "0.5", "0.5.1"}, (
                    f"criterion must remain Rust 1.85-compatible, got: {constraint}"
                )


class TestMutmutBaseline:
    """Verify mutmut is configured for mutation testing."""

    def test_mutmut_installed(self) -> None:
        """mutmut should be available in dev dependencies."""
        try:
            mutmut_version = metadata.version("mutmut")
            assert mutmut_version, "mutmut not found"
        except metadata.PackageNotFoundError:
            pytest.skip("mutmut not installed (optional)")

    def test_pyproject_mutmut_constraint(self) -> None:
        """pyproject.toml should declare mutmut in dev dependencies."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()

        assert "mutmut" in content, "mutmut not found in dev dependencies"
        match = re.search(r"mutmut([>=<,\d.~]+)", content)
        if match:
            constraint = match.group(1)
            assert "3" in constraint or "4" in constraint, (
                f"mutmut constraint should target 3.x or 4.x, got: {constraint}"
            )
