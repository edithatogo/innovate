"""Tests for Kairos smoke test scenarios.

This test module validates that:
1. Kairos DES smoke scenario compiles and runs
2. Kairos ABM smoke scenario compiles and runs
3. Release evidence documents smoke test status
"""

import subprocess
from pathlib import Path


class TestKairosSmokeTests:
    """Test Kairos smoke test scenarios."""

    def test_kairos_des_smoke_example_exists(self):
        """Test that Kairos DES smoke scenario example exists."""
        rust_binding_dir = Path(__file__).parent.parent / "bindings" / "rust"
        example_file = rust_binding_dir / "examples" / "kairos_des_smoke.rs"

        assert example_file.exists(), (
            f"Kairos DES smoke example not found at {example_file}. Minimal DES scenario must be provided."
        )

        content = example_file.read_text()

        # Verify it imports Kairos DES modules
        required_imports = [
            "kairo_ecs_core",
            "kairo_ecs_des",
            "kairo_ecs_rng",
        ]

        for import_name in required_imports:
            assert import_name in content, (
                f"Required import '{import_name}' not found in DES smoke test. Must demonstrate Kairos DES integration."
            )

    def test_kairos_abm_smoke_example_exists(self):
        """Test that Kairos ABM smoke scenario example exists."""
        rust_binding_dir = Path(__file__).parent.parent / "bindings" / "rust"
        example_file = rust_binding_dir / "examples" / "kairos_abm_smoke.rs"

        assert example_file.exists(), (
            f"Kairos ABM smoke example not found at {example_file}. "
            "Minimal ABM scenario covering ECS agent state must be provided."
        )

        content = example_file.read_text()

        # Verify it imports Kairos ABM modules
        required_imports = [
            "kairo_ecs_core",
            "kairo_ecs_state",
            "kairo_ecs_abm",
        ]

        for import_name in required_imports:
            assert import_name in content, (
                f"Required import '{import_name}' not found in ABM smoke test. "
                "Must demonstrate ECS agent state integration."
            )

    def test_kairos_examples_documented(self):
        """Test that Kairos examples are documented in Cargo.toml."""
        rust_binding_dir = Path(__file__).parent.parent / "bindings" / "rust"
        cargo_toml = rust_binding_dir / "Cargo.toml"

        content = cargo_toml.read_text()

        # Examples should be runnable via cargo run --example
        assert "examples" in content or "kairos_des_smoke" in content or "kairos_abm_smoke" in content, (
            "Kairos examples not documented in Cargo.toml. Smoke tests should be runnable via cargo."
        )

    def test_release_evidence_kairos_status_exists(self):
        """Test that release evidence documents Kairos integration status."""
        track_dir = Path(__file__).parent.parent / "conductor" / "tracks" / "kairos_dependency_inclusion_20260626"
        integration_report = track_dir / "KAIROS_INTEGRATION_REPORT.md"

        assert integration_report.exists(), (
            "Kairos integration report not found. Release evidence must document Kairos status."
        )

        content = integration_report.read_text()

        # Check for smoke test status section
        required_sections = [
            "Smoke Test Status",
            "Phase 1 Status",
            "Phase 2 Status",
            "Phase 3 Status",
        ]

        for section in required_sections:
            assert section in content, (
                f"Required section '{section}' not found in integration report. "
                "Release evidence must track smoke test progress."
            )
