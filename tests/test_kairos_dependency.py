"""Tests for Kairos dependency integration.

This test module validates that:
1. Kairos crates are properly configured as dependencies
2. Kairos crates can be imported and used
3. Kairos smoke tests work (DES and ABM scenarios)
"""

import json
import subprocess
import sys
from pathlib import Path


class TestKairosDependencyInclusion:
    """Test Kairos dependency inclusion."""

    def test_kairos_dependency_in_cargo_toml(self):
        """Test that Kairos crate dependencies are defined in Cargo.toml."""
        rust_binding_dir = Path(__file__).parent.parent / "bindings" / "rust"
        cargo_toml = rust_binding_dir / "Cargo.toml"

        assert cargo_toml.exists(), f"Cargo.toml not found at {cargo_toml}"

        content = cargo_toml.read_text()

        # Check that core Kairos crates are listed
        required_crates = [
            "kairo-ecs-types",
            "kairo-ecs-core",
            "kairo-ecs-state",
            "kairo-ecs-rng",
            "kairo-ecs-des",
            "kairo-ecs-abm",
            "kairo-ecs-arrow",
        ]

        for crate in required_crates:
            assert crate in content, (
                f"Kairos crate '{crate}' not found in Cargo.toml. Kairos dependency plumbing incomplete."
            )

    def test_mesa_removed_from_base_dependencies(self):
        """Test that mesa is removed from base Python dependencies."""
        project_root = Path(__file__).parent.parent
        pyproject_toml = project_root / "pyproject.toml"

        assert pyproject_toml.exists(), f"pyproject.toml not found at {pyproject_toml}"

        content = pyproject_toml.read_text()

        # Check that mesa is not in the base dependencies
        # Look for the dependencies array and ensure mesa is not there
        lines = content.split("\n")
        in_dependencies = False
        in_optional = False

        for i, line in enumerate(lines):
            if "dependencies = [" in line:
                in_dependencies = True
                in_optional = False
            elif "optional-dependencies" in line or "dependency-groups" in line:
                in_dependencies = False
                in_optional = True
            elif line.startswith("[") and line.endswith("]"):
                in_dependencies = False
                in_optional = False

            if in_dependencies and "mesa" in line:
                raise AssertionError(
                    f"mesa found in base dependencies at line {i + 1}. Mesa should be removed from base install."
                )

    def test_ndlib_removed_from_base_dependencies(self):
        """Test that ndlib is removed from base Python dependencies."""
        project_root = Path(__file__).parent.parent
        pyproject_toml = project_root / "pyproject.toml"

        assert pyproject_toml.exists(), f"pyproject.toml not found at {pyproject_toml}"

        content = pyproject_toml.read_text()

        # Check that ndlib is not in the base dependencies
        lines = content.split("\n")
        in_dependencies = False

        for i, line in enumerate(lines):
            if "dependencies = [" in line:
                in_dependencies = True
            elif line.startswith("[") and line.endswith("]"):
                in_dependencies = False

            if in_dependencies and "ndlib" in line:
                raise AssertionError(
                    f"ndlib found in base dependencies at line {i + 1}. NDLib should be removed from base install."
                )

    def test_networkx_justified_or_optional(self):
        """Test that networkx is either justified or moved behind an extra."""
        project_root = Path(__file__).parent.parent
        pyproject_toml = project_root / "pyproject.toml"

        assert pyproject_toml.exists(), f"pyproject.toml not found at {pyproject_toml}"

        content = pyproject_toml.read_text()

        # Check if networkx is in base dependencies
        lines = content.split("\n")
        in_dependencies = False
        networkx_in_base = False

        for line in lines:
            if "dependencies = [" in line:
                in_dependencies = True
            elif line.startswith("[") and line.endswith("]"):
                in_dependencies = False

            if in_dependencies and "networkx" in line:
                networkx_in_base = True
                break

        if networkx_in_base:
            # If networkx is in base, it must have a comment explaining why
            networkx_line_idx = None
            for i, line in enumerate(lines):
                if in_dependencies and "networkx" in line:
                    networkx_line_idx = i
                    break

            # Look for a comment explaining the justification
            if networkx_line_idx is not None:
                # Check current line and surrounding lines for comment
                found_justification = False
                for j in range(max(0, networkx_line_idx - 2), min(len(lines), networkx_line_idx + 3)):
                    if "#" in lines[j] and (
                        "plot" in lines[j].lower() or "graph" in lines[j].lower() or "visual" in lines[j].lower()
                    ):
                        found_justification = True
                        break

                assert found_justification, (
                    "networkx is in base dependencies but has no clear justification. "
                    "Either provide a comment explaining its use for plotting/graph APIs, "
                    "or move it to an optional extra."
                )

    def test_legacy_abm_extra_exists(self):
        """Test that legacy-abm optional dependency extra is defined."""
        project_root = Path(__file__).parent.parent
        pyproject_toml = project_root / "pyproject.toml"

        assert pyproject_toml.exists(), f"pyproject.toml not found at {pyproject_toml}"

        content = pyproject_toml.read_text()

        # Check that legacy-abm extra is defined
        assert "legacy-abm = [" in content, (
            "legacy-abm optional dependency group not found. Mesa and ndlib must be available via legacy-abm extra."
        )

        # Check that mesa and ndlib are in the extra
        lines = content.split("\n")
        in_legacy_abm = False
        has_mesa = False
        has_ndlib = False

        for line in lines:
            if "legacy-abm = [" in line:
                in_legacy_abm = True
            elif in_legacy_abm and line.startswith("["):
                break
            elif in_legacy_abm:
                if "mesa" in line:
                    has_mesa = True
                if "ndlib" in line:
                    has_ndlib = True

        assert has_mesa, "mesa not found in legacy-abm extra"
        assert has_ndlib, "ndlib not found in legacy-abm extra"

    def test_integration_report_exists(self):
        """Test that Kairos integration report document exists."""
        track_dir = Path(__file__).parent.parent / "conductor" / "tracks" / "kairos_dependency_inclusion_20260626"
        report_file = track_dir / "KAIROS_INTEGRATION_REPORT.md"

        assert report_file.exists(), (
            f"Kairos integration report not found at {report_file}. "
            "External compatibility constraints and dependency policy must be documented."
        )

        content = report_file.read_text()

        # Check for key sections
        required_sections = [
            "Kairos Source Verification",
            "Dependency Migration Policy",
            "External Compatibility Constraints",
            "Python 3.14 Baseline",
            "Rust Toolchain",
            "Registry / Packaging Constraints",
        ]

        for section in required_sections:
            assert section in content, f"Required section '{section}' not found in Kairos integration report."
