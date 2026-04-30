"""Contract tests for standardized diagnostics and uncertainty reporting."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from innovate.backend import use_backend
from innovate.diffuse.bass import BassModel
from innovate.fitters import BootstrapFitter, DiagnosticsContract
from innovate.fitters import DiagnosticsWarning as FittersDiagnosticsWarning
from innovate.fitters.diagnostics_contract import (
    DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION,
    DiagnosticsArtifactPayload,
    DiagnosticsWarning,
    UncertaintySummary,
    build_diagnostics_contract,
)
from innovate.fitters.scipy_fitter import ScipyFitter
from innovate.utils.model_evaluation import compare_models


class TestDiagnosticsContract:
    """Tests for the shared diagnostics contract."""

    def setup_method(self) -> None:
        """Use the NumPy backend for deterministic contract tests."""
        use_backend("numpy")

    def _fit_bass_model(self) -> tuple[BassModel, np.ndarray, np.ndarray]:
        t = np.linspace(1, 12, 20)
        model = BassModel()
        model.params_ = {"p": 0.03, "q": 0.35, "m": 1000.0}
        y = model.predict(t)
        fitter = ScipyFitter()
        fitter.fit(BassModel(), t, y)
        fitted_model = BassModel()
        fitter.fit(fitted_model, t, y)
        return fitted_model, t, y

    def test_fit_diagnostics_exposes_contract_fields(self) -> None:
        """Fit diagnostics should expose warnings and uncertainty metadata."""
        model, t, y = self._fit_bass_model()

        contract = build_diagnostics_contract(model, t, y)

        assert contract.support_level == "supported"
        assert contract.uncertainty.report_type == "point_estimate"
        assert contract.uncertainty.provenance == "deterministic"
        assert contract.residual_analysis is not None
        assert contract.warnings == []
        assert contract.metrics["RMSE"] >= 0
        assert contract.metrics["R_squared"] == contract.metrics["R-squared"]

    def test_supported_uncertainty_variants_round_trip(self) -> None:
        """The contract should support deterministic, bootstrap, and posterior uncertainty."""
        point = UncertaintySummary.point_estimate(provenance="deterministic")
        bootstrap = UncertaintySummary.bootstrap_interval(
            lower={"p": 0.1, "q": 0.2},
            upper={"p": 0.4, "q": 0.5},
            median={"p": 0.25, "q": 0.35},
            level=0.95,
        )
        posterior = UncertaintySummary.posterior_summary(
            lower={"p": 0.11},
            upper={"p": 0.43},
            median={"p": 0.27},
            level=0.9,
        )

        assert point.report_type == "point_estimate"
        assert point.support_level == "supported"
        assert bootstrap.report_type == "bootstrap_interval"
        assert bootstrap.provenance == "bootstrap"
        assert bootstrap.lower["p"] == 0.1
        assert posterior.report_type == "posterior_summary"
        assert posterior.provenance == "bayesian"
        assert posterior.level == 0.9

    def test_unsupported_model_surface_is_explicit(self) -> None:
        """The contract should report unsupported diagnostics explicitly."""

        class BrokenModel:
            param_names: ClassVar[list[str]] = []
            params_: ClassVar[dict[str, float]] = {}

        contract = build_diagnostics_contract(BrokenModel(), [1.0, 2.0], [1.0, 2.0])

        assert contract.support_level == "unsupported"
        assert contract.uncertainty.report_type == "unsupported"
        assert any(w.code == "model_unavailable" for w in contract.warnings)

    def test_compare_models_annotates_diagnostics_contract(self) -> None:
        """Model comparison should expose the standardized diagnostics metadata."""
        model, t, y = self._fit_bass_model()
        comparison_df = compare_models({"Bass": model}, t, y)

        assert "Diagnostics Support" in comparison_df.columns
        assert "Uncertainty Report Type" in comparison_df.columns
        assert "Warning Count" in comparison_df.columns
        assert comparison_df.loc["Bass", "Diagnostics Support"] == "supported"
        assert comparison_df.loc["Bass", "Uncertainty Report Type"] == "point_estimate"

    def test_diagnostics_warning_serialization(self) -> None:
        """Diagnostics warnings should be serializable and preserve metadata."""
        warning = DiagnosticsWarning(code="optimizer_message", message="converged with warning")
        payload = warning.to_dict()

        assert payload == {
            "code": "optimizer_message",
            "message": "converged with warning",
            "severity": "warning",
        }

    def test_fitters_package_exports_contract_types(self) -> None:
        """The fitters package should re-export the diagnostics contract types."""
        assert DiagnosticsContract.__name__ == "DiagnosticsContract"
        assert FittersDiagnosticsWarning.__name__ == "DiagnosticsWarning"

    def test_contract_serialization_is_json_friendly(self) -> None:
        """The canonical contract should serialize arrays and analysis objects."""
        model, t, y = self._fit_bass_model()
        contract = build_diagnostics_contract(model, t, y)

        payload = contract.to_dict()

        assert isinstance(payload["residuals"], list)
        assert isinstance(payload["uncertainty"]["samples"], dict)
        assert payload["residual_analysis"] is not None
        assert isinstance(payload["residual_analysis"]["residuals"], list)

    def test_contract_serialization_includes_versioned_artifacts(self) -> None:
        """Diagnostics should expose a stable artifact payload for bindings."""
        model, t, y = self._fit_bass_model()
        contract = build_diagnostics_contract(model, t, y, model_name="BassModel")

        payload = contract.to_dict()["artifacts"]

        assert payload["schema_version"] == DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION
        assert payload["model_name"] == "BassModel"
        assert payload["backend"] == "numpy"
        assert payload["xla"]["eligible"] is False
        assert payload["artifacts"]["residuals"]["kind"] == "residual_diagnostics"
        assert payload["artifacts"]["residuals"]["columns"] == (
            "index",
            "residual",
            "standardized_residual",
        )
        assert payload["artifacts"]["residuals"]["rows"][0]["index"] == 0
        assert payload["artifacts"]["calibration"]["kind"] == "calibration_check"
        assert payload["artifacts"]["model_comparison"]["kind"] == "model_comparison"

    def test_diagnostics_artifact_payload_emits_kernel_table_payloads(self) -> None:
        """Tabular diagnostics artifacts should be usable through kernel table payloads."""
        model, t, y = self._fit_bass_model()
        contract = build_diagnostics_contract(model, t, y, model_name="BassModel")
        artifact = DiagnosticsArtifactPayload.from_contract(contract)

        tables = artifact.to_table_payloads()

        assert set(tables) == {"model_comparison", "residuals", "uncertainty"}
        assert tables["residuals"].columns == ("index", "residual", "standardized_residual")
        assert tables["residuals"].metadata["diagnostics_artifact_kind"] == "residual_diagnostics"
        assert tables["model_comparison"].columns == ("metric", "value")

    def test_compare_models_keeps_unsupported_models_explicit(self) -> None:
        """Model comparison should not attempt to infer metrics for unsupported models."""

        class UnsupportedModel:
            def predict(self, t: np.ndarray) -> np.ndarray:
                return np.asarray(t, dtype=float)

            param_names: ClassVar[list[str]] = []
            params_: ClassVar[dict[str, float]] = {}

        comparison_df = compare_models({"unsupported": UnsupportedModel()}, [1.0, 2.0], [1.0, 2.0])

        assert comparison_df.loc["unsupported", "Diagnostics Support"] == "unsupported"
        assert comparison_df.loc["unsupported", "Warning Count"] >= 1
        assert "Parameters" in comparison_df.columns

    def test_compare_models_logs_when_predict_is_missing(self, caplog) -> None:
        """Model comparison should use logging for missing predict methods."""

        class MissingPredictModel:
            param_names: ClassVar[list[str]] = []
            params_: ClassVar[dict[str, float]] = {}

        caplog.set_level("WARNING", logger="innovate.utils.model_evaluation")

        comparison_df = compare_models(
            {"missing_predict": MissingPredictModel()},
            [1.0, 2.0],
            [1.0, 2.0],
        )

        assert comparison_df.empty
        assert any("does not have a 'predict' method" in record.message for record in caplog.records)

    def test_bootstrap_fitter_reports_explicit_unsupported_uncertainty(self) -> None:
        """Bootstrap fitters should surface an explicit unsupported uncertainty marker before fitting."""
        fitter = BootstrapFitter(n_bootstraps=3)
        model = BassModel()

        contract = fitter.get_diagnostics_contract(model, [1.0, 2.0], [1.0, 2.0])

        assert contract.support_level == "unsupported"
        assert contract.uncertainty.report_type == "unsupported"
        assert contract.uncertainty.provenance == "bootstrap"
        assert any(w.code == "bootstrap_unavailable" for w in contract.warnings)
