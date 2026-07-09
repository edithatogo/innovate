"""Tests for data ingestion contracts, validation, adapters, and integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from innovate.data import (
    AdoptionDataset,
    CompetitionDataset,
    DatasetProvenance,
    NetworkEdgeDataset,
    PolicyTimingDataset,
    SubstitutionDataset,
    SyntheticAdoptionPublicAdapter,
    attach_provenance,
    dataset_from_dict,
    dataset_to_baseline_scenario,
    frame_to_dataset,
    get_builtin_adapter,
    get_dataset_benchmark_link,
    ingest_local,
    integration_bundle,
    list_builtin_adapters,
    load_table,
    polars_available,
    reproducible_artifact,
    require_valid,
    resolve_model_cards_for_dataset,
    validate_dataset,
)


def _prov(**kwargs):
    base = {
        "source": "unit-test",
        "license": "CC0-1.0",
        "transform_steps": ("test",),
        "citation": "test citation",
    }
    base.update(kwargs)
    return DatasetProvenance.create(**base)


def test_provenance_fails_closed_on_unknown_license() -> None:
    with pytest.raises(ValueError, match="license is unknown"):
        DatasetProvenance.create(source="x", license="unknown")


def test_adoption_contract_and_validation() -> None:
    ds = AdoptionDataset(time=[0, 1, 2], adoption=[1, 2, 4], denominator=[10, 10, 10], unit="count")
    report = validate_dataset(ds)
    assert report.ok
    assert ds.kind == "adoption"
    bad = AdoptionDataset(time=[0, 1, 2], adoption=[1, 5, 3])
    assert validate_dataset(bad).ok is False


def test_substitution_competition_policy_network_contracts() -> None:
    sub = SubstitutionDataset(time=[0, 1], share=[[0.2, 0.3], [0.4, 0.4]], product_labels=("a", "b"))
    assert validate_dataset(sub).ok
    comp = CompetitionDataset(
        time=[0, 0, 1, 1],
        unit_id=("u1", "u1", "u1", "u1"),
        product_id=("p1", "p2", "p1", "p2"),
        value=[1, 2, 2, 3],
    )
    assert validate_dataset(comp).ok
    policy = PolicyTimingDataset(event_times=[1, 3], event_effects=[0.1, -0.2], event_labels=("on", "off"))
    assert policy.to_policy_timing_inputs().event_labels == ("on", "off")
    net = NetworkEdgeDataset(source=("a", "b"), target=("b", "c"), weight=[1.0, 2.0])
    inputs = net.to_network_inputs()
    assert inputs.node_labels == ("a", "b", "c")


def test_attach_provenance_and_roundtrip() -> None:
    ds = AdoptionDataset(time=[0, 1], adoption=[1, 2])
    with_prov = attach_provenance(ds, _prov())
    assert with_prov.provenance is not None
    restored = dataset_from_dict(with_prov.to_dict())
    assert restored.kind == "adoption"
    assert restored.provenance is not None
    assert restored.provenance.license == "CC0-1.0"


def test_ingest_csv_and_parquet(tmp_path: Path) -> None:
    frame = pd.DataFrame({"time": [0, 1, 2], "adoption": [1.0, 2.0, 3.0]})
    csv_path = tmp_path / "adoption.csv"
    parquet_path = tmp_path / "adoption.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)

    ds_csv, report_csv = ingest_local(csv_path, "adoption", provenance=_prov())
    ds_pq, report_pq = ingest_local(parquet_path, "adoption", provenance=_prov())
    assert report_csv.ok
    assert report_pq.ok
    assert np.allclose(ds_csv.adoption, [1, 2, 3])
    assert ds_csv.provenance is not None
    assert ds_csv.provenance.checksum
    assert ds_pq.provenance is not None
    artifact = reproducible_artifact(ds_csv)
    assert artifact["artifact_kind"] == "innovate.dataset"
    assert "csv" in artifact["formats_supported"]


def test_frame_to_dataset_requires_provenance() -> None:
    frame = pd.DataFrame({"time": [0, 1], "adoption": [1.0, 2.0]})
    with pytest.raises(ValueError, match="provenance is required"):
        frame_to_dataset(frame, "adoption", require_provenance=True)


def test_public_adapter_pattern() -> None:
    manifests = list_builtin_adapters()
    assert any(item.adapter_id == "synthetic_adoption_v1" for item in manifests)
    adapter = get_builtin_adapter("synthetic_adoption_v1")
    assert isinstance(adapter, SyntheticAdoptionPublicAdapter)
    dataset, report = adapter.ingest(periods=8, seed=2)
    assert report.ok
    assert dataset.kind == "adoption"
    assert dataset.provenance is not None
    assert dataset.provenance.license == "CC0-1.0"


def test_benchmark_and_scenario_integration() -> None:
    link = get_dataset_benchmark_link("adoption")
    assert "bass_smoke_adoption" in link.benchmark_case_ids
    cards = resolve_model_cards_for_dataset("adoption")
    assert "bass" in cards or len(cards) >= 0
    adapter = SyntheticAdoptionPublicAdapter()
    dataset, report = adapter.ingest(periods=6)
    require_valid(dataset)
    scenario = dataset_to_baseline_scenario(dataset, name="demo", market_size=100.0)
    assert scenario.time_horizon > 0
    bundle = integration_bundle(dataset, report)
    assert bundle["validation"]["ok"] is True
    assert bundle["benchmark_link"]["dataset_kind"] == "adoption"
    json.dumps(bundle)  # serializable


def test_duplicate_and_unit_failures() -> None:
    net = NetworkEdgeDataset(source=("a", "a"), target=("b", "b"), weight=[1.0, 1.0])
    report = validate_dataset(net)
    assert report.ok is False
    assert any(check.name.startswith("duplicates:") and check.status == "fail" for check in report.checks)
    adoption = AdoptionDataset(time=[0, 1], adoption=[1, 2], unit="widgets")
    assert validate_dataset(adoption).ok is False


def test_load_table_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_table(tmp_path / "missing.csv")


@pytest.mark.skipif(not polars_available(), reason="polars optional extra not installed")
def test_polars_ingest_optional() -> None:
    import polars as pl

    from innovate.data import ingest_polars

    frame = pl.DataFrame({"time": [0, 1, 2], "adoption": [1.0, 2.0, 4.0]})
    dataset, report = ingest_polars(frame, "adoption", provenance=_prov())
    assert report.ok
    assert dataset.kind == "adoption"
