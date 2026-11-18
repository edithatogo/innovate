"""Check frozen public-data semantics and temporal holdout isolation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "docs/datasets/wdi-aus-internet-20261116"


@pytest.fixture(scope="module")
def example():
    """Load the maintained example while retaining its real native fit function."""
    spec = importlib.util.spec_from_file_location("native_public_data", ROOT / "examples/native_public_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dataset_copy(tmp_path):
    """Give mutation tests their own copy of the frozen public extract."""
    directory = tmp_path / DATASET.name
    shutil.copytree(DATASET, directory)
    return directory


def update_manifest(directory: Path, **changes) -> None:
    """Change fixture metadata without modifying the checked-in dataset."""
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def update_file_digest(directory: Path, filename: str) -> None:
    """Allow semantic mutation tests to get past the separate checksum guard."""
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"][filename] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_calendar_and_population_percentages_remain_observed(example):
    frame, manifest = example.load_observations()
    assert frame["year"].to_list() == list(range(1990, 2026))
    assert frame["time"].to_list() == [float(index) for index in range(36)]
    assert frame.filter(pl.col("observed").is_null())["year"].to_list() == [2002, 2003, 2004, 2025]
    assert frame.filter(pl.col("year") == 1995)["observed"][0] == 2.76
    assert frame.filter(pl.col("year") == 2005)["time"][0] == 15.0
    assert manifest["unit"] == "percent_of_population"
    assert manifest["time_unit"] == "year"
    assert manifest["observed_rows"] == 32


def test_corrupted_frozen_bytes_fail_before_fitting(example, dataset_copy):
    with (dataset_copy / "observations.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        example.load_observations(dataset_copy)


def test_frozen_dataset_survives_git_checkout_with_crlf_defaults(example, tmp_path):
    """Exercise actual Git checkout conversion while retaining hashed data bytes."""
    repository = tmp_path / "checkout"
    repository.mkdir()
    git = ["git", "-c", "core.autocrlf=true", "-c", "core.safecrlf=false", "-C", str(repository)]
    subprocess.run([*git, "init", "--quiet", "--template="], check=True, capture_output=True, timeout=30)
    relative = DATASET.relative_to(ROOT)
    checked_out = repository / relative
    checked_out.parent.mkdir(parents=True)
    shutil.copytree(DATASET, checked_out)
    attributes = DATASET.parent / ".gitattributes"
    if attributes.exists():
        shutil.copyfile(attributes, checked_out.parent / attributes.name)
    # The control proves CRLF conversion is active even when the snapshot opts out.
    control = repository / "ordinary.txt"
    control.write_bytes(b"first\nsecond\n")
    subprocess.run([*git, "add", "--all"], check=True, capture_output=True, timeout=30)
    for path in [control, *checked_out.iterdir()]:
        path.unlink()
    subprocess.run([*git, "checkout-index", "--all", "--force"], check=True, capture_output=True, timeout=30)
    assert control.read_bytes() == b"first\r\nsecond\r\n"
    frame, manifest = example.load_observations(checked_out)
    assert frame.height == 36 and manifest["observed_rows"] == 32
    for name in ("manifest.json", "raw.json", "observations.csv"):
        assert (checked_out / name).read_bytes() == (DATASET / name).read_bytes()


@pytest.mark.parametrize(
    "old,new",
    [("2002,12,\n", "2002,12,0\n"), ("2005,15,63\n", "2005,12,63\n"), ("1995,5,2.76\n", "1995,5,0.0276\n")],
)
def test_null_fill_compressed_time_and_percent_rescaling_are_rejected(example, dataset_copy, old, new):
    path = dataset_copy / "observations.csv"
    original = path.read_text(encoding="utf-8")
    assert old in original
    path.write_text(original.replace(old, new), encoding="utf-8")
    update_file_digest(dataset_copy, "observations.csv")
    with pytest.raises(ValueError, match="transformation"):
        example.load_observations(dataset_copy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("country_iso3", "NZL"),
        ("indicator", "IT.CEL.SETS.P2"),
        ("unit", "proportion"),
        ("time_origin", 2000),
        ("license", "unknown"),
        ("time_unit", "month"),
    ],
)
def test_changed_observation_identity_or_units_fail(example, dataset_copy, field, value):
    update_manifest(dataset_copy, **{field: value})
    with pytest.raises(ValueError):
        example.load_observations(dataset_copy)


def test_different_raw_country_is_rejected_even_after_rehash(example, dataset_copy):
    path = dataset_copy / "raw.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw[1][0]["countryiso3code"] = "NZL"
    path.write_text(json.dumps(raw), encoding="utf-8")
    update_file_digest(dataset_copy, "raw.json")
    with pytest.raises(ValueError, match="country or indicator"):
        example.load_observations(dataset_copy)


@pytest.mark.parametrize("cutoff", [True, 2009, 2021, 2015.0, "2015"])
def test_cutoff_requires_a_supported_integer_year(example, cutoff):
    with pytest.raises(ValueError, match="integer training cutoff"):
        example.run_comparison(cutoff)


def test_held_out_values_do_not_change_fitted_parameters_or_predictions(example):
    frame, manifest = example.load_observations()
    original = example.run_comparison(2015)
    changed = frame.with_columns(
        pl.when((pl.col("year") > 2015) & pl.col("observed").is_not_null())
        .then(pl.col("observed") - 2.0)
        .otherwise(pl.col("observed"))
        .alias("observed")
    )
    # Replace observations only at the comparison boundary; the separate loader
    # tests enforce frozen-data provenance. Both calls still fit the real core.
    with patch.object(example, "load_observations", return_value=(changed, manifest)):
        alternative = example.run_comparison(2015)
    assert original["parameters"] == alternative["parameters"]
    assert [row["prediction"] for row in original["predictions"]] == [
        row["prediction"] for row in alternative["predictions"]
    ]
    assert [row["observed"] for row in original["predictions"]] != [
        row["observed"] for row in alternative["predictions"]
    ]
    assert original["scores"] != alternative["scores"]
    assert {row["year"] for row in original["predictions"]} == set(range(2016, 2025))


def test_persistence_score_uses_training_endpoint_and_percentage_point_errors(example):
    result = example.run_comparison(2015)
    baseline = [row for row in result["predictions"] if row["model"] == "persistence"]
    assert {row["prediction"] for row in baseline} == {84.5605149091408}
    score = next(row for row in result["scores"] if row["model"] == "persistence")
    assert score["holdout_mae"] == pytest.approx(8.410123487267734)
    assert score["holdout_bias"] == pytest.approx(-score["holdout_mae"])
    assert score["holdout_rmse"] >= score["holdout_mae"]
    assert result["training_rows"] == 23 and result["holdout_rows"] == 9
