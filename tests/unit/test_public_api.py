"""Tests for the canonical public API surface."""

import pytest

import innovate
from innovate import backends, benchmarks, remote_execution
from innovate.backends.numpy_backend import NumPyBackend
from innovate.base import DiffusionModel
from innovate.benchmarks import BenchmarkAutomationReport, BenchmarkCase, BenchmarkFamily, ModelCard
from innovate.capabilities import ModelCapability
from innovate.compete.competition import MultiProductDiffusionModel as StableCompetitionModel
from innovate.compete.lotka_volterra import LotkaVolterraModel
from innovate.diffuse.bass import BassModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.diffuse.logistic import LogisticModel
from innovate.ecosystem.complementary_goods import ComplementaryGoodsModel
from innovate.fitters import ScipyFitter
from innovate.remote_execution import InProcessRemoteExecutor, RemoteExecutionRequest
from innovate.substitute.composite import CompositeDiffusionModel
from innovate.substitute.fisher_pry import FisherPryModel
from innovate.substitute.norton_bass import NortonBassModel


def test_top_level_exports_resolve_to_canonical_classes():
    """The package root should expose stable model and fitter classes."""
    assert innovate.DiffusionModel is DiffusionModel
    assert innovate.BassModel is BassModel
    assert innovate.LogisticModel is LogisticModel
    assert innovate.GompertzModel is GompertzModel
    assert innovate.FisherPryModel is FisherPryModel
    assert innovate.NortonBassModel is NortonBassModel
    assert innovate.CompositeDiffusionModel is CompositeDiffusionModel
    assert innovate.MultiProductDiffusionModel is StableCompetitionModel
    assert innovate.LotkaVolterraModel is LotkaVolterraModel
    assert innovate.ComplementaryGoodsModel is ComplementaryGoodsModel
    assert innovate.ScipyFitter is ScipyFitter
    assert innovate.BenchmarkAutomationReport is BenchmarkAutomationReport
    assert innovate.BenchmarkCase is BenchmarkCase
    assert innovate.BenchmarkFamily is BenchmarkFamily
    assert innovate.BenchmarkJob is benchmarks.BenchmarkJob
    assert innovate.BenchmarkRunner is benchmarks.BenchmarkRunner
    assert innovate.BenchmarkSuiteResult is benchmarks.BenchmarkSuiteResult
    assert innovate.InProcessRemoteExecutor is InProcessRemoteExecutor
    assert innovate.RemoteExecutionRequest is RemoteExecutionRequest
    assert innovate.ModelCard is ModelCard


def test_canonical_subpackages_export_stable_models():
    """Stable model families should have explicit package-level exports."""
    assert innovate.diffuse.BassModel is BassModel
    assert innovate.diffuse.LogisticModel is LogisticModel
    assert innovate.diffuse.GompertzModel is GompertzModel
    assert innovate.substitute.FisherPryModel is FisherPryModel
    assert innovate.substitute.NortonBassModel is NortonBassModel
    assert innovate.substitute.CompositeDiffusionModel is CompositeDiffusionModel
    assert innovate.compete.MultiProductDiffusionModel is StableCompetitionModel
    assert innovate.compete.LotkaVolterraModel is LotkaVolterraModel
    assert innovate.ecosystem.ComplementaryGoodsModel is ComplementaryGoodsModel
    assert benchmarks.BenchmarkCase is BenchmarkCase
    assert benchmarks.BenchmarkFamily is BenchmarkFamily
    assert benchmarks.BenchmarkJob is innovate.BenchmarkJob
    assert benchmarks.BenchmarkRunner is innovate.BenchmarkRunner
    assert benchmarks.BenchmarkSuiteResult is innovate.BenchmarkSuiteResult
    assert benchmarks.validate_benchmark_corpus is innovate.validate_benchmark_corpus
    assert benchmarks.refresh_model_card_summaries is innovate.refresh_model_card_summaries
    assert benchmarks.ModelCard is ModelCard
    assert remote_execution.InProcessRemoteExecutor is innovate.InProcessRemoteExecutor


def test_legacy_paths_remain_importable_while_canonical_paths_exist():
    """Existing import paths should continue to resolve during topology cleanup."""
    from innovate import backend as legacy_backend
    from innovate.compete.competition import MultiProductDiffusionModel as LegacyCompetitionModel

    assert legacy_backend.use_backend is innovate.backend.use_backend
    assert LegacyCompetitionModel is innovate.compete.MultiProductDiffusionModel


def test_model_capability_registry_exposes_stable_families():
    """Stable models should be discoverable from the canonical registry."""
    registry = innovate.get_model_registry()

    assert set(registry) >= {
        "bass",
        "logistic",
        "gompertz",
        "fisher_pry",
        "norton_bass",
        "multi_product",
        "lotka_volterra",
    }
    assert isinstance(registry["bass"], ModelCapability)
    assert registry["bass"].import_path == "innovate.diffuse.BassModel"
    assert registry["multi_product"].supports_multivariate_output is True
    assert innovate.get_model_capability("lotka_volterra").family == "competition"

    with pytest.raises(KeyError, match="Unknown model capability"):
        innovate.get_model_capability("does_not_exist")


def test_backends_namespace_forwards_runtime_backend_state():
    """The canonical plural namespace should mirror the runtime selector."""
    innovate.backend.use_backend("numpy")

    assert backends.use_backend is innovate.backend.use_backend
    assert isinstance(backends.current_backend, NumPyBackend)
