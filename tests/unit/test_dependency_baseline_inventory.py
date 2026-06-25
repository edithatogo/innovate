"""Tests for dependency baseline inventory script."""
import json
import subprocess
from pathlib import Path

import pytest

INVENTORY_SCRIPT = Path("scripts/dependency_baseline_inventory.py")
INVENTORY_OUTPUT = Path("docs/source/_static/dependency_baseline_inventory.json")


class TestDependencyBaselineInventory:
    """Verify the dependency inventory script produces valid output."""

    def test_inventory_script_exists(self):
        """Red phase: ensure the inventory script is in place."""
        assert INVENTORY_SCRIPT.exists(), (
            f"Inventory script not found at {INVENTORY_SCRIPT}"
        )

    def test_inventory_runs_and_produces_json(self):
        """Red phase: script must execute and produce a JSON report."""
        result = subprocess.run(
            ["uv", "run", "python", str(INVENTORY_SCRIPT)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"Inventory script failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
        )
        assert INVENTORY_OUTPUT.exists(), (
            f"Inventory output not found at {INVENTORY_OUTPUT}"
        )

    def test_inventory_json_structure(self):
        """Red phase: the JSON report must have expected top-level keys."""
        assert INVENTORY_OUTPUT.exists(), "Run inventory script first"
        data = json.loads(INVENTORY_OUTPUT.read_text())
        assert "track" in data
        assert "phase" in data
        assert "task" in data
        assert "python" in data
        assert data["track"] == "bleeding-edge-dependency-modernization"

    def test_inventory_python_outdated_count(self):
        """Red phase: there should be outdated Python dependencies reported."""
        data = json.loads(INVENTORY_OUTPUT.read_text())
        assert data["python"]["count"] > 0, (
            "Expected at least one outdated Python dependency"
        )

    def test_inventory_frontend_exists(self):
        """Red phase: frontend outdated checks should be present."""
        data = json.loads(INVENTORY_OUTPUT.read_text())
        assert "frontend" in data
        # Should either have data or an error key
        assert "raw" in data["frontend"] or "error" in data["frontend"]

    def test_inventory_rust_deps_reported(self):
        """Red phase: Rust Cargo.toml dependencies should be inventoried."""
        data = json.loads(INVENTORY_OUTPUT.read_text())
        if "deps" in data["rust"]:
            assert len(data["rust"]["deps"]) > 0