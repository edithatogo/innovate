"""Link validated datasets to benchmarks, model cards, and scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from innovate.benchmarks.model_cards import ModelCard, get_model_card, list_model_cards
from innovate.data.contracts import DatasetContract, DatasetKind
from innovate.data.validation import ValidationReport, require_valid
from innovate.scenario.schemas import BaselineScenario


@dataclass(frozen=True, slots=True)
class DatasetBenchmarkLink:
    """Association between a dataset kind and benchmark/model-card targets."""

    dataset_kind: DatasetKind
    benchmark_case_ids: tuple[str, ...]
    model_card_keys: tuple[str, ...]
    scenario_types: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_kind": self.dataset_kind,
            "benchmark_case_ids": list(self.benchmark_case_ids),
            "model_card_keys": list(self.model_card_keys),
            "scenario_types": list(self.scenario_types),
        }


_DATASET_BENCHMARK_LINKS: dict[DatasetKind, DatasetBenchmarkLink] = {
    "adoption": DatasetBenchmarkLink(
        dataset_kind="adoption",
        benchmark_case_ids=("bass_smoke_adoption", "logistic_growth_smoke"),
        model_card_keys=("bass", "logistic", "gompertz"),
        scenario_types=("baseline", "intervention"),
    ),
    "substitution": DatasetBenchmarkLink(
        dataset_kind="substitution",
        benchmark_case_ids=("fisher_pry_replacement_smoke",),
        model_card_keys=("fisher_pry", "norton_bass"),
        scenario_types=("substitution",),
    ),
    "competition": DatasetBenchmarkLink(
        dataset_kind="competition",
        benchmark_case_ids=("lock_in_smoke",),
        model_card_keys=("lock_in",),
        scenario_types=("competition",),
    ),
    "policy_timing": DatasetBenchmarkLink(
        dataset_kind="policy_timing",
        benchmark_case_ids=("bass_smoke_adoption",),
        model_card_keys=("bass",),
        scenario_types=("intervention",),
    ),
    "network_edges": DatasetBenchmarkLink(
        dataset_kind="network_edges",
        benchmark_case_ids=("bass_smoke_adoption",),
        model_card_keys=("bass",),
        scenario_types=("network",),
    ),
}


def get_dataset_benchmark_link(kind: DatasetKind) -> DatasetBenchmarkLink:
    """Return benchmark/model-card links for a dataset kind."""
    try:
        return _DATASET_BENCHMARK_LINKS[kind]
    except KeyError as exc:
        raise KeyError(f"no benchmark link for dataset kind: {kind}") from exc


def list_dataset_benchmark_links() -> dict[str, DatasetBenchmarkLink]:
    """Return all dataset-to-benchmark link records."""
    return dict(_DATASET_BENCHMARK_LINKS)


def resolve_model_cards_for_dataset(kind: DatasetKind) -> dict[str, ModelCard]:
    """Resolve available model cards linked to a dataset kind."""
    link = get_dataset_benchmark_link(kind)
    cards = list_model_cards()
    resolved: dict[str, ModelCard] = {}
    for key in link.model_card_keys:
        if key in cards:
            resolved[key] = cards[key]
        else:
            try:
                resolved[key] = get_model_card(key)
            except KeyError:
                continue
    return resolved


def dataset_to_baseline_scenario(
    dataset: DatasetContract,
    *,
    name: str,
    description: str = "",
    time_unit: str = "years",
    reference_year: int = 2026,
    market_size: float = 100.0,
) -> BaselineScenario:
    """Create a baseline scenario summary from an adoption-like dataset.

    Only adoption datasets with time/adoption series are supported for this
    convenience helper; other kinds should use dedicated scenario builders.
    """
    require_valid(dataset)
    if dataset.kind != "adoption":
        raise TypeError("dataset_to_baseline_scenario currently supports adoption datasets only")
    time = dataset.time  # type: ignore[attr-defined]
    adoption = dataset.adoption  # type: ignore[attr-defined]
    horizon = float(time[-1] - time[0]) if len(time) > 1 else float(time[0] or 1.0)
    if horizon <= 0:
        horizon = float(len(time))
    initial = float(adoption[0] / market_size) if market_size else 0.0
    initial = min(1.0, max(0.0, initial))
    return BaselineScenario(
        name=name,
        description=description or f"Baseline derived from {dataset.kind} dataset",
        time_horizon=horizon,
        time_unit=time_unit,
        reference_year=reference_year,
        market_size=market_size,
        initial_adoption=initial,
    )


def integration_bundle(dataset: DatasetContract, report: ValidationReport) -> dict[str, Any]:
    """Build a release-friendly bundle linking dataset, validation, and cards."""
    require_valid(dataset)
    link = get_dataset_benchmark_link(dataset.kind)
    cards = {key: card.to_dict() for key, card in resolve_model_cards_for_dataset(dataset.kind).items()}
    return {
        "dataset": dataset.to_dict(),
        "validation": report.to_dict(),
        "benchmark_link": link.to_dict(),
        "model_cards": cards,
        "provenance": None if dataset.provenance is None else dataset.provenance.to_dict(),
    }
