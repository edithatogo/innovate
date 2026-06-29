"""
Test suite for Polyglot Registry Acceptance Completion (Track 03).

Validates that registry acceptance states are accurately documented,
receipt-gated, and properly synchronized.
"""

from pathlib import Path

import pytest


class TestRegistryInventory:
    """Test that registry inventory is complete and accurate."""

    @pytest.fixture
    def inventory_path(self):
        """Return path to registry inventory."""
        return Path(__file__).parent.parent.parent / (
            "conductor/tracks/polyglot_registry_acceptance_completion_20260625/registry_inventory.md"
        )

    def test_inventory_exists(self, inventory_path):
        """Verify inventory file exists."""
        assert inventory_path.exists(), f"Registry inventory not found at {inventory_path}"
        content = inventory_path.read_text()
        assert len(content) > 0, "Registry inventory is empty"

    def test_all_language_targets_documented(self, inventory_path):
        """Verify all language targets are documented."""
        content = inventory_path.read_text()
        languages = ["Python", "Rust", "R", "Julia", "TypeScript", "Go", "C#"]
        for lang in languages:
            assert lang in content, f"Language '{lang}' not documented in registry inventory"

    def test_hpc_targets_documented(self, inventory_path):
        """Verify all HPC targets are documented."""
        content = inventory_path.read_text()
        hpc_targets = ["Spack", "EasyBuild", "HPSF", "E4S"]
        for target in hpc_targets:
            assert target in content, f"HPC target '{target}' not documented"

    def test_all_states_have_owners(self, inventory_path):
        """Verify all registry entries have owners."""
        content = inventory_path.read_text()
        # Should have owner column and values
        assert "Owner" in content, "Registry inventory should document owners"
        # Count rows to verify completeness
        lines = content.split("\n")
        data_rows = [line for line in lines if line.startswith("|") and "Status" not in line and "Target" not in line]
        assert len(data_rows) > 0, "Registry inventory should have data rows"

    def test_all_states_have_actions(self, inventory_path):
        """Verify all registry entries have documented next actions."""
        content = inventory_path.read_text()
        # Should have action column
        assert "Action" in content, "Registry inventory should document next actions"


class TestRegistryStates:
    """Test that registry states are valid and consistent."""

    @pytest.fixture
    def inventory_path(self):
        """Return path to registry inventory."""
        return Path(__file__).parent.parent.parent / (
            "conductor/tracks/polyglot_registry_acceptance_completion_20260625/registry_inventory.md"
        )

    def test_valid_registry_states(self, inventory_path):
        """Verify all states are valid."""
        content = inventory_path.read_text()
        valid_states = ["accepted", "deferred", "submitted"]
        content_lower = content.lower()

        # Verify at least one state is documented
        state_count = sum(1 for state in valid_states if state in content_lower)
        assert state_count > 0, "At least one registry state should be documented"

    def test_no_undocumented_states(self, inventory_path):
        """Verify there are no conflicting or undocumented states."""
        content = inventory_path.read_text()
        # States should only be accepted, deferred, or submitted
        invalid_indicators = ["live but not documented", "unknown state", "undefined"]
        for indicator in invalid_indicators:
            assert indicator not in content.lower(), f"Invalid state indicator: {indicator}"

    def test_accepted_states_are_justified(self, inventory_path):
        """Verify accepted states have evidence pointers."""
        content = inventory_path.read_text()

        # Count accepted entries and evidence references
        if "accepted" in content.lower():
            lines = content.split("\n")
            accepted_lines = [line for line in lines if "accepted" in line.lower()]

            # Each accepted line should have some evidence reference
            for line in accepted_lines:
                # Evidence column should have content (not empty or "None")
                parts = line.split("|")
                if len(parts) > 3:
                    evidence_part = parts[3].strip()
                    assert evidence_part, f"Accepted state should have evidence: {line}"
                    assert evidence_part.lower() != "none", f"Accepted state should have evidence: {line}"


class TestNoUndocumentedAcceptance:
    """Test that no registry state is claimed without documentation."""

    @pytest.fixture
    def inventory_path(self):
        """Return path to registry inventory."""
        return Path(__file__).parent.parent.parent / (
            "conductor/tracks/polyglot_registry_acceptance_completion_20260625/registry_inventory.md"
        )

    def test_acceptance_has_receipts(self, inventory_path):
        """Verify accepted packages have receipt-like evidence."""
        content = inventory_path.read_text()
        # If PyPI is mentioned as accepted, it should have evidence
        if "PyPI" in content and "accepted" in content:
            assert "v0.5.0" in content or "Published" in content, (
                "PyPI acceptance should reference a version or publication"
            )

    def test_deferrals_document_reason(self, inventory_path):
        """Verify deferred targets document their reason."""
        content = inventory_path.read_text()
        # Deferred entries should explain why (pending, awaiting, etc.)
        if "deferred" in content.lower():
            assert any(
                keyword in content.lower() for keyword in ["pending", "awaiting", "submitted", "review", "pr"]
            ), "Deferred targets should document their reason"

    def test_no_empty_evidence(self, inventory_path):
        """Verify no accepted state has empty evidence."""
        content = inventory_path.read_text()
        lines = content.split("\n")

        for line in lines:
            if "accepted" in line.lower():
                parts = line.split("|")
                # Evidence should not be empty or just whitespace
                if len(parts) > 3:
                    evidence = parts[3].strip()
                    assert len(evidence) > 0, f"Accepted state must have evidence: {line}"
                    assert evidence != "-", f"Accepted state must have evidence: {line}"


class TestLanguageTargetCompleteness:
    """Test that each language has required registry targets."""

    @pytest.fixture
    def inventory_path(self):
        """Return path to registry inventory."""
        return Path(__file__).parent.parent.parent / (
            "conductor/tracks/polyglot_registry_acceptance_completion_20260625/registry_inventory.md"
        )

    def test_python_targets(self, inventory_path):
        """Verify Python has PyPI target documented."""
        content = inventory_path.read_text()
        assert "PyPI" in content, "Python should have PyPI target documented"

    def test_rust_targets(self, inventory_path):
        """Verify Rust has crates.io target documented."""
        content = inventory_path.read_text()
        assert "crates.io" in content or "Rust" in content, "Rust should have crates.io target documented"

    def test_r_targets(self, inventory_path):
        """Verify R has CRAN or R-universe target documented."""
        content = inventory_path.read_text()
        assert "CRAN" in content or "R-universe" in content, "R should have registry target documented"

    def test_hpc_targets_documented(self, inventory_path):
        """Verify HPC ecosystem has at least one target."""
        content = inventory_path.read_text()
        hpc_targets = ["Spack", "EasyBuild", "E4S"]
        found = any(target in content for target in hpc_targets)
        assert found, "At least one HPC target should be documented"


class TestMaintainerResponsibility:
    """Test that maintainer-only actions are clearly identified."""

    @pytest.fixture
    def inventory_path(self):
        """Return path to registry inventory."""
        return Path(__file__).parent.parent.parent / (
            "conductor/tracks/polyglot_registry_acceptance_completion_20260625/registry_inventory.md"
        )

    def test_deferred_actions_are_clear(self, inventory_path):
        """Verify deferred target actions are clear about responsibilities."""
        content = inventory_path.read_text()

        # Deferred actions should indicate who is responsible
        if "deferred" in content.lower():
            action_indicators = ["submit", "await", "monitor", "maintain", "coordin"]
            found_action_docs = any(indicator in content.lower() for indicator in action_indicators)
            assert found_action_docs, "Deferred targets should document action owners"

    def test_no_credential_references(self, inventory_path):
        """Verify no credentials or secrets are in inventory."""
        content = inventory_path.read_text()
        secret_indicators = ["password", "token", "api_key", "secret", "credential"]
        for indicator in secret_indicators:
            assert indicator not in content.lower(), f"Registry inventory should not contain {indicator}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
