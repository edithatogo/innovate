"""Tests for Kairos Rust/toolchain build plumbing.

This test module validates that:
1. Kairos crates are properly integrated in the build system
2. Rust builds succeed with Kairos crates
3. Cargo.lock is properly updated
"""

import subprocess
from pathlib import Path


class TestKairosBuildPlumbing:
    """Test Kairos Rust/toolchain build integration."""

    def test_rust_bindings_cargo_lock_exists(self):
        """Test that Cargo.lock exists in Rust bindings."""
        rust_binding_dir = Path(__file__).parent.parent / "bindings" / "rust"
        cargo_lock = rust_binding_dir / "Cargo.lock"

        assert cargo_lock.exists(), (
            f"Cargo.lock not found at {cargo_lock}. Lock file must be tracked for reproducible builds."
        )

    def test_rust_bindings_cargo_lock_includes_kairos(self):
        """Test that Cargo.lock includes Kairos crate entries."""
        rust_binding_dir = Path(__file__).parent.parent / "bindings" / "rust"
        cargo_lock = rust_binding_dir / "Cargo.lock"

        assert cargo_lock.exists(), f"Cargo.lock not found at {cargo_lock}"

        content = cargo_lock.read_text()

        # Check that Kairos crates are in the lock file
        required_crates = [
            'name = "kairo-ecs-types"',
            'name = "kairo-ecs-core"',
            'name = "kairo-ecs-state"',
            'name = "kairo-ecs-rng"',
            'name = "kairo-ecs-des"',
            'name = "kairo-ecs-abm"',
            'name = "kairo-ecs-arrow"',
        ]

        for crate in required_crates:
            assert crate in content, (
                f"Kairos crate entry '{crate}' not found in Cargo.lock. "
                "Lock file must be updated to include Kairos dependencies."
            )

    def test_rust_bindings_cargo_lock_has_kairos_revision(self):
        """Test that Cargo.lock records the Kairos repository revision."""
        rust_binding_dir = Path(__file__).parent.parent / "bindings" / "rust"
        cargo_lock = rust_binding_dir / "Cargo.lock"

        assert cargo_lock.exists(), f"Cargo.lock not found at {cargo_lock}"

        content = cargo_lock.read_text()

        # Check for the specific revision we're using
        expected_revision = "fae901558f07b7b717a676adbafbe2cdc78dea1c"

        assert expected_revision in content, (
            f"Kairos revision '{expected_revision}' not found in Cargo.lock. "
            "Lock file must record the exact revision for reproducibility."
        )

    def test_rust_bindings_build_succeeds(self):
        """Test that Rust bindings build successfully with Kairos crates."""
        rust_binding_dir = Path(__file__).parent.parent / "bindings" / "rust"

        # Run cargo check to verify the build system works
        result = subprocess.run(
            ["cargo", "check"],
            cwd=rust_binding_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"Rust build failed with exit code {result.returncode}. stderr: {result.stderr}\nstdout: {result.stdout}"
        )

        # Verify that build succeeded (may be cached, so just check for success messages)
        output = result.stdout + result.stderr
        assert "Finished" in output or "Checking" in output, (
            f"Build output does not contain expected messages. Output: {output}"
        )
