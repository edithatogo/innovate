"""Benchmark execution harness for stable model families."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from innovate.base.base import DiffusionModel
from innovate.fitters import ScipyFitter
from innovate.fitters.diagnostics_contract import DiagnosticsContract, UncertaintySummary, build_diagnostics_contract

from .corpus import BenchmarkCase, list_benchmark_cases
from .model_cards import get_model_card, list_model_cards

_RUNNABLE_STABLE_MODEL_KEYS: tuple[str, ...] = (
    "bass",
    "logistic",
    "gompertz",
    "fisher_pry",
    "norton_bass",
    "multi_product",
)


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """Serialized output from running a model against a benchmark case."""

    case_id: str
    model_key: str
    model_name: str
    family: str
    predictions: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: DiagnosticsContract = field(default_factory=DiagnosticsContract)
    uncertainty: UncertaintySummary = field(default_factory=UncertaintySummary.point_estimate)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the benchmark run to a JSON-friendly dictionary."""
        return {
            "case_id": self.case_id,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "family": self.family,
            "predictions": self.predictions.tolist(),
            "metrics": self.metrics,
            "diagnostics": self.diagnostics.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "metadata": self.metadata,
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the run artifact to disk as JSON."""
        output_path = Path(path)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return output_path


@dataclass(frozen=True, slots=True)
class BenchmarkJob:
    """Canonical job pairing a stable model family with one benchmark case."""

    model_key: str
    case_id: str
    model_name: str
    family: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model_key": self.model_key,
            "case_id": self.case_id,
            "model_name": self.model_name,
            "family": self.family,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    """Serialized output from running a benchmark suite."""

    runs: tuple[BenchmarkRun, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_count": len(self.runs),
            "runs": [run.to_dict() for run in self.runs],
        }

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return output_path


def _resolve_model_class(import_path: str) -> type[DiffusionModel]:
    module_name, class_name = import_path.rsplit(".", 1)
    module = import_module(module_name)
    model_cls = getattr(module, class_name)
    return model_cls


def _build_model_instance(model_key: str) -> DiffusionModel:
    if model_key == "multi_product":
        from innovate.compete.competition import MultiProductDiffusionModel

        return MultiProductDiffusionModel(
            p=[0.03, 0.02],
            Q=[[0.28, 0.06], [0.04, 0.24]],
            m=[700.0, 300.0],
            names=["Product A", "Product B"],
        )

    card = get_model_card(model_key)
    model_cls = _resolve_model_class(card.import_path)
    return model_cls()


def list_benchmark_jobs(model_keys: tuple[str, ...] | None = None) -> tuple[BenchmarkJob, ...]:
    """Return the stable benchmark jobs in canonical order."""
    cards = list_model_cards()
    selected_keys = model_keys or _RUNNABLE_STABLE_MODEL_KEYS
    jobs: list[BenchmarkJob] = []
    for model_key in selected_keys:
        if model_key not in cards:
            raise KeyError(f"Unknown benchmark model key: {model_key}")
        card = cards[model_key]
        for case_id in card.benchmark_case_ids:
            jobs.append(
                BenchmarkJob(
                    model_key=model_key,
                    case_id=case_id,
                    model_name=card.model_name,
                    family=card.family,
                ),
            )
    return tuple(jobs)


def run_stable_benchmark_suite(
    *,
    runner: BenchmarkRunner | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> BenchmarkSuiteResult:
    """Run the stable benchmark suite for the selected model families."""
    suite_runner = runner or BenchmarkRunner()
    results: list[BenchmarkRun] = []
    case_by_id = {case.case_id: case for case in list_benchmark_cases()}

    for job in list_benchmark_jobs(model_keys=model_keys):
        case = case_by_id[job.case_id]
        results.append(
            suite_runner.run(
                _build_model_instance(job.model_key),
                case,
                model_key=job.model_key,
            ),
        )

    return BenchmarkSuiteResult(runs=tuple(results))


class BenchmarkRunner:
    """Canonical runner for executing a model against a benchmark case."""

    def __init__(self, fitter: Any | None = None):
        self.fitter = fitter or ScipyFitter()

    def _ensure_model(self, model: DiffusionModel | type[DiffusionModel]) -> DiffusionModel:
        if isinstance(model, type):
            return model()
        return model

    def run(
        self,
        model: DiffusionModel | type[DiffusionModel],
        case: BenchmarkCase,
        *,
        model_key: str | None = None,
        fitter: Any | None = None,
    ) -> BenchmarkRun:
        """Fit a model on a benchmark case and return a structured result."""
        model_obj = self._ensure_model(model)
        fitter_obj = fitter or self.fitter

        fitter_obj.fit(model_obj, case.time, case.observed)

        predictions = np.asarray(model_obj.predict(case.time), dtype=float)
        diagnostics = build_diagnostics_contract(
            model_obj,
            case.time,
            case.observed,
            model_name=model_obj.__class__.__name__,
        )

        return BenchmarkRun(
            case_id=case.case_id,
            model_key=model_key or case.canonical_model_key,
            model_name=model_obj.__class__.__name__,
            family=case.family.value,
            predictions=predictions,
            metrics=diagnostics.metrics,
            diagnostics=diagnostics,
            uncertainty=diagnostics.uncertainty,
            metadata=dict(case.metadata),
        )
