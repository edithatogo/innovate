"""Real Voiage normalized bundles at Innovate's installed Polars boundary."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
import voiage
from voiage.contracts import normalized_input
from voiage.contracts.normalized_input import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
)


@pytest.fixture(scope="module")
def qualifier():
    path = Path(__file__).resolve().parents[2] / "scripts" / "qualify_voiage_arrow.py"
    spec = importlib.util.spec_from_file_location("voiage_qualification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle(table: pa.Table, *, descriptor_digest: str = "a" * 64) -> NormalizedInputBundle:
    return NormalizedInputBundle(
        manifest=DatasetManifest(
            dataset_id="innovate-normalized-qualification",
            tables=(
                TableManifest(
                    table_id="samples",
                    fields=tuple(FieldManifest(field_id=field.name, dtype=str(field.type)) for field in table.schema),
                ),
            ),
            provenance=SourceProvenance(
                provider_id="synthetic-test",
                source_uri="file:///redacted/innovate-consumer",
                descriptor_digest=descriptor_digest,
            ),
        ),
        tables={"samples": table},
    )


def _write(bundle: NormalizedInputBundle, directory: Path, format: str) -> Path:
    path = directory / ("samples.arrow" if format == "ipc" else "samples.parquet")
    getattr(bundle, f"write_{format}")(path)
    return path


@pytest.mark.parametrize("format", ["ipc", "parquet"])
def test_consumes_normalized_values_without_changing_row_order(qualifier, tmp_path, format):
    bundle = _bundle(pa.table({"time": [0.0, 1.0, 2.0], "value": [2.0, 3.5, 4.0]}))
    path = _write(bundle, tmp_path, format)

    frame = qualifier.read_normalized_frame(
        path, format=format, table_id="samples", expected_digest=bundle.content_digest
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ["time", "value"]
    assert frame.rows() == [(0.0, 2.0), (1.0, 3.5), (2.0, 4.0)]


@pytest.mark.parametrize("format", ["ipc", "parquet"])
def test_multi_chunk_round_trip_preserves_digest_and_consumed_values(qualifier, tmp_path, format):
    table = pa.table(
        {
            "time": pa.chunked_array([[0.0, 1.0], [2.0]]),
            "value": pa.chunked_array([[2.0, 3.5], [4.0]]),
        }
    )
    bundle = _bundle(table)
    path = _write(bundle, tmp_path, format)
    restored = getattr(NormalizedInputBundle, f"read_{format}")(path)

    assert [[len(chunk) for chunk in column.chunks] for column in restored.table("samples").columns] == [[2, 1], [2, 1]]
    assert restored.content_digest == bundle.content_digest
    frame = qualifier.read_normalized_frame(
        path, format=format, table_id="samples", expected_digest=bundle.content_digest
    )
    assert frame.rows() == [(0.0, 2.0), (1.0, 3.5), (2.0, 4.0)]


@pytest.mark.parametrize("format", ["ipc", "parquet"])
def test_restores_timestamp_unit_and_nested_field_metadata_before_consumption(qualifier, tmp_path, format):
    covariance_type = pa.list_(pa.field("covariance_element", pa.float64(), metadata={b"unit": b"AUD^2"}), 2)
    table = pa.table(
        {
            "at": pa.array([0, 1], type=pa.timestamp("s")),
            "covariance": pa.array([[1.0, 0.25], [0.25, 4.0]], type=covariance_type),
        }
    ).replace_schema_metadata({b"fixture": b"synthetic-normalized-types"})
    bundle = _bundle(table)
    path = _write(bundle, tmp_path, format)
    restored = getattr(NormalizedInputBundle, f"read_{format}")(path)

    assert restored.table("samples").schema.equals(table.schema, check_metadata=True)
    assert restored.content_digest == bundle.content_digest
    frame = qualifier.read_normalized_frame(
        path, format=format, table_id="samples", expected_digest=bundle.content_digest
    )
    assert frame["at"].to_list() == [dt.datetime(1970, 1, 1), dt.datetime(1970, 1, 1, 0, 0, 1)]
    assert frame["covariance"].to_list() == [[1.0, 0.25], [0.25, 4.0]]


@pytest.mark.parametrize("format", ["ipc", "parquet"])
@pytest.mark.parametrize("mutation", ["value", "provenance"])
def test_rejects_changed_input_identity_before_polars_conversion(qualifier, tmp_path, monkeypatch, format, mutation):
    original = _bundle(pa.table({"value": [1.0, 2.0]}))
    changed = (
        _bundle(pa.table({"value": [1.0, 3.0]}))
        if mutation == "value"
        else _bundle(original.table("samples"), descriptor_digest="b" * 64)
    )
    path = _write(changed, tmp_path, format)
    assert changed.content_digest != original.content_digest

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError("unverified normalized input reached Polars conversion")

    monkeypatch.setattr(pl, "from_arrow", unexpected_conversion)
    with pytest.raises(ValueError, match="digest"):
        qualifier.read_normalized_frame(
            path, format=format, table_id="samples", expected_digest=original.content_digest
        )


@pytest.mark.parametrize("format", ["ipc", "parquet"])
def test_rejects_missing_explicit_table(qualifier, tmp_path, format):
    bundle = _bundle(pa.table({"value": [1.0]}))
    path = _write(bundle, tmp_path, format)

    with pytest.raises(ValueError, match="table"):
        qualifier.read_normalized_frame(path, format=format, table_id="missing", expected_digest=bundle.content_digest)


@pytest.mark.parametrize("format", ["ipc", "parquet"])
def test_rejects_ordinary_files_without_normalized_metadata(qualifier, tmp_path, format):
    table = pa.table({"value": [1.0]})
    path = tmp_path / "ordinary"
    if format == "ipc":
        with ipc.new_file(path, table.schema) as writer:
            writer.write_table(table)
    else:
        pq.write_table(table, path)

    with pytest.raises(ValueError, match="normalized"):
        qualifier.read_normalized_frame(path, format=format, table_id="samples", expected_digest="a" * 64)


def test_rejects_unsupported_format_without_reading_the_file(qualifier, tmp_path):
    with pytest.raises(ValueError, match="format"):
        qualifier.read_normalized_frame(
            tmp_path / "not-created.csv", format="csv", table_id="samples", expected_digest="a" * 64
        )


def _installed_package_archive(tmp_path, *, include_contract=True, corrupt_init=False):
    """Make a bounded verification fixture from real installed package files."""
    init = Path(voiage.__file__).resolve()
    contract = Path(normalized_input.__file__).resolve()
    wheel = tmp_path / "package-members.whl"
    with ZipFile(wheel, "w") as archive:
        contents = init.read_bytes()
        if corrupt_init:
            contents += b"\n# altered package verification fixture\n"
        archive.writestr("voiage/__init__.py", contents)
        if include_contract:
            archive.write(contract, "voiage/contracts/normalized_input.py")
    return wheel, [init, contract]


def test_wheel_verification_requires_imported_module_members(qualifier, tmp_path):
    wheel, imported_paths = _installed_package_archive(tmp_path, include_contract=False)

    with pytest.raises(ValueError, match="member|missing|supplied wheel"):
        qualifier.verify_installed_wheel("voiage", wheel, imported_paths)


def test_wheel_verification_rejects_changed_package_bytes(qualifier, tmp_path):
    wheel, imported_paths = _installed_package_archive(tmp_path, corrupt_init=True)

    with pytest.raises(ValueError, match="differs|mismatch"):
        qualifier.verify_installed_wheel("voiage", wheel, imported_paths)


def test_wheel_verification_hashes_matching_installed_package_files(qualifier, tmp_path):
    wheel, imported_paths = _installed_package_archive(tmp_path)

    assert (
        qualifier.verify_installed_wheel("voiage", wheel, imported_paths)
        == hashlib.sha256(wheel.read_bytes()).hexdigest()
    )


def test_failed_installed_run_retains_nonzero_status_and_isolated_command(qualifier, tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", qualifier.__file__, "--wheel", str(tmp_path / "missing.whl")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "failed"
    assert receipt["exit_status"] == result.returncode
    assert receipt["command"][1] == "-I"
    assert not Path(receipt["working_directory"]).is_relative_to(receipt["source_checkout"])
    assert "normalized_cases" not in receipt
