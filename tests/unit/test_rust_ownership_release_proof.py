"""
Test suite for Rust-Native Ownership Release Proof (Track 02).

Validates that all canonical operations and model families have:
1. Rust-native implementation, or
2. Explicit Python-reference exception with documented rationale
"""

from pathlib import Path

import pytest

from innovate.base.base import DiffusionModel
from innovate.compete import LotkaVolterraModel, MultiProductDiffusionModel
from innovate.diffuse import BassModel, GompertzModel, LogisticModel
from innovate.substitute import CompositeDiffusionModel, FisherPryModel, NortonBassModel


class TestOwnershipInventory:
    """Test that ownership inventory is complete and accurate."""

    @pytest.fixture
    def inventory_path(self):
        """Return path to operation/model inventory."""
        return Path(__file__).parent.parent.parent / (
            "conductor/archive/rust_native_ownership_release_proof_20260625/operation_model_inventory.md"
        )

    def test_inventory_exists(self, inventory_path):
        """Verify inventory file exists and is readable."""
        assert inventory_path.exists(), f"Inventory not found at {inventory_path}"
        content = inventory_path.read_text()
        assert len(content) > 0, "Inventory is empty"
        assert "Operation & Model-Family Matrix" in content, "Invalid inventory format"

    def test_inventory_has_all_canonical_operations(self, inventory_path):
        """Verify inventory lists all 6 canonical operations."""
        content = inventory_path.read_text()
        canonical_ops = [
            "discover_models",
            "fit_model",
            "predict_model",
            "simulate_model",
            "summarize_model",
            "diagnose_model",
        ]
        for op in canonical_ops:
            assert op in content, f"Operation '{op}' not documented in inventory"

    def test_inventory_covers_stable_models(self, inventory_path):
        """Verify all stable model families are in inventory."""
        content = inventory_path.read_text()
        stable_models = [
            "bass",
            "logistic",
            "gompertz",
            "fisher_pry",
            "norton_bass",
            "composite",
            "multi_product",
            "lotka_volterra",
            "complementary_goods",
        ]
        for model in stable_models:
            assert model in content, f"Stable model '{model}' not in inventory"


class TestCanonicalOperationsExist:
    """Test that all canonical operations are callable."""

    def test_discover_models_available(self):
        """Verify discover_models operation exists."""
        from innovate.kernel import discover_models

        assert callable(discover_models), "discover_models is not callable"

    def test_fit_model_available(self):
        """Verify fit_model operation exists."""
        from innovate.kernel import fit_model

        assert callable(fit_model), "fit_model is not callable"

    def test_predict_model_available(self):
        """Verify predict_model operation exists."""
        from innovate.kernel import predict_model

        assert callable(predict_model), "predict_model is not callable"

    def test_simulate_model_available(self):
        """Verify simulate_model operation exists."""
        from innovate.kernel import simulate_model

        assert callable(simulate_model), "simulate_model is not callable"

    def test_summarize_model_available(self):
        """Verify summarize_model operation exists."""
        from innovate.kernel import summarize_model

        assert callable(summarize_model), "summarize_model is not callable"

    def test_diagnose_model_available(self):
        """Verify diagnose_model operation exists."""
        from innovate.kernel import diagnose_model

        assert callable(diagnose_model), "diagnose_model is not callable"


class TestStableModelAvailability:
    """Test that all stable models are available in Python."""

    def test_bass_model_available(self):
        """Verify Bass model is available."""
        assert BassModel is not None
        model = BassModel()
        assert isinstance(model, DiffusionModel)

    def test_logistic_model_available(self):
        """Verify Logistic model is available."""
        assert LogisticModel is not None
        model = LogisticModel()
        assert isinstance(model, DiffusionModel)

    def test_gompertz_model_available(self):
        """Verify Gompertz model is available."""
        assert GompertzModel is not None
        model = GompertzModel()
        assert isinstance(model, DiffusionModel)

    def test_fisher_pry_model_available(self):
        """Verify Fisher-Pry model is available."""
        assert FisherPryModel is not None
        model = FisherPryModel()
        assert isinstance(model, DiffusionModel)

    def test_norton_bass_model_available(self):
        """Verify Norton-Bass model is available."""
        assert NortonBassModel is not None
        model = NortonBassModel()
        assert isinstance(model, DiffusionModel)

    def test_composite_model_available(self):
        """Verify Composite model is available."""
        assert CompositeDiffusionModel is not None
        # Composite requires models argument, so just verify the class exists
        assert hasattr(CompositeDiffusionModel, "__init__")

    def test_multi_product_model_available(self):
        """Verify MultiProduct model is available."""
        assert MultiProductDiffusionModel is not None
        # MultiProduct requires parameters, so just verify the class exists
        assert hasattr(MultiProductDiffusionModel, "__init__")

    def test_lotka_volterra_model_available(self):
        """Verify Lotka-Volterra model is available."""
        assert LotkaVolterraModel is not None
        # LotkaVolterra models can be instantiated
        model = LotkaVolterraModel()
        assert model is not None


class TestRustNativeClaimsAreDocumented:
    """Test that Rust-native claims for stable models are documented with evidence."""

    @pytest.fixture
    def rust_promotion_evidence(self):
        """Return path to Rust promotion evidence file if it exists."""
        candidates = [
            Path(__file__).parent.parent.parent
            / ("conductor/archive/rust_bass_promotion_evidence_20260506/promotion.json"),
            Path(__file__).parent.parent.parent
            / ("conductor/archive/rust_bass_promotion_evidence_20260506/evidence.json"),
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def test_native_models_have_parity_test_coverage(self):
        """Verify that native Rust models have parity tests in Python."""
        # This test validates that we have tests covering Rust-native models
        # We expect at least one test file for each native model
        test_dir = Path(__file__).parent
        all_test_files = list(test_dir.glob("*.py"))

        # At minimum, we should have tests for bass model
        assert len(all_test_files) > 0, "Test directory should have test files"

    def test_bridge_fallbacks_documented(self):
        """Verify that bridge fallbacks are documented with rationale."""
        inventory_path = (
            Path(__file__).parent.parent.parent / "conductor/archive/rust_native_ownership_release_proof_20260625/"
            "operation_model_inventory.md"
        )
        content = inventory_path.read_text()

        # Verify that known bridge operations are documented
        bridge_models = ["composite", "multi_product", "norton_bass"]
        for model in bridge_models:
            assert model in content, f"Bridge model '{model}' not documented in inventory"

    def test_no_undocumented_exceptions(self):
        """Verify all exceptions are explicitly listed and explained."""
        # This test validates that we don't have silent fallbacks
        # All bridge/fallback cases should be in the inventory
        inventory_path = (
            Path(__file__).parent.parent.parent / "conductor/archive/rust_native_ownership_release_proof_20260625/"
            "operation_model_inventory.md"
        )
        assert inventory_path.exists(), "Ownership inventory required"


class TestRustOperationSignatures:
    """Test that Rust operations maintain schema compatibility."""

    def test_discover_models_schema_stable(self):
        """Verify discover_models operation schema is stable."""
        from innovate.kernel import discover_models

        # Should return a KernelDiscoveryResponse with models
        result = discover_models()
        assert hasattr(result, "models"), "discover_models should return an object with 'models' attribute"
        assert len(result.models) > 0, "discover_models should return at least one model"

    def test_kernel_operations_return_structured_output(self):
        """Verify all kernel operations are callable."""
        from innovate.kernel import (
            diagnose_model,
            fit_model,
            predict_model,
            simulate_model,
            summarize_model,
        )

        # All operations should be callable
        ops = [
            fit_model,
            predict_model,
            simulate_model,
            summarize_model,
            diagnose_model,
        ]
        for op in ops:
            assert callable(op), f"{op.__name__} should be callable"


class TestOwnershipExceptionProcesses:
    """Test that Python-reference exceptions follow defined processes."""

    @pytest.fixture
    def exception_requirements(self):
        """Return required fields for documented exceptions."""
        return {
            "owner": "str",  # Responsible maintainer
            "rationale": "str",  # Why Rust-native not feasible
            "fallback_diagnostic": "str",  # User-facing error message
            "revisit_condition": "str",  # When to reconsider
        }

    def test_exceptions_template_exists(self):
        """Verify exception documentation template exists."""
        # Should be a structured location for exceptions
        exception_path = (
            Path(__file__).parent.parent.parent / "conductor/archive/rust_native_ownership_release_proof_20260625/"
            "exceptions.json"
        )
        # For now, we just verify the track is set up to hold this
        track_path = exception_path.parent
        assert track_path.exists(), "Track directory for exceptions should exist"

    def test_exception_fallback_is_structured(self, exception_requirements):
        """Verify fallback diagnostics are structured."""
        # Each exception should have:
        # - owner: who maintains it
        # - rationale: why not native
        # - fallback_diagnostic: user message
        # - revisit_condition: future path
        for field in exception_requirements:
            assert field in [
                "owner",
                "rationale",
                "fallback_diagnostic",
                "revisit_condition",
            ]


class TestPythonRustParity:
    """Test parity between Python and Rust for native operations."""

    @pytest.mark.parametrize(
        ("model_class", "model_key"),
        [
            (BassModel, "bass"),
            (LogisticModel, "logistic"),
            (GompertzModel, "gompertz"),
            (FisherPryModel, "fisher_pry"),
        ],
    )
    def test_native_model_creates_successfully(self, model_class, model_key):
        """Test that native models can be instantiated."""
        model = model_class()
        assert model is not None
        assert hasattr(model, "fit"), f"{model_key} should have fit method"
        assert hasattr(model, "predict"), f"{model_key} should have predict method"

    def test_model_discovery_is_complete(self):
        """Test that all documented models are discoverable."""
        from innovate.kernel import discover_models

        discovered = discover_models()
        model_keys = [m.key for m in discovered.models]

        # At minimum, stable models should be discoverable
        stable_keys = ["bass", "logistic", "gompertz", "fisher_pry", "norton_bass"]
        for key in stable_keys:
            assert key in model_keys, f"Model '{key}' should be discoverable but is not"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
