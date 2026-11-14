"""Installed native Innovate → Polars/Arrow → Voiage consumer qualification."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from importlib.metadata import distribution, version
from importlib.util import find_spec
from math import exp, isclose, log
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import polars as pl
import pyarrow as pa
import voiage
from voiage.contracts import normalized_input
from voiage.contracts.normalized_input import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
)
from voiage.methods.basic import evpi
from voiage.schema import ValueArray


def read_normalized_frame(path: Path, *, format: str, table_id: str, expected_digest: str) -> pl.DataFrame:
    """Verify a Voiage bundle's identity before crossing the Polars boundary.

    The expected digest comes from the producer, separately from the file.
    Voiage's digest includes Arrow metadata and chunk layout; converting first
    would discard evidence needed to verify the original input.
    """
    readers = {"ipc": NormalizedInputBundle.read_ipc, "parquet": NormalizedInputBundle.read_parquet}
    if format not in readers:
        raise ValueError("normalized input format must be ipc or parquet")
    bundle = readers[format](path)
    if bundle.content_digest != expected_digest:
        raise ValueError("normalized input content_digest mismatch")
    if table_id not in bundle.tables:
        raise ValueError("normalized input table is absent")
    return pl.from_arrow(bundle.table(table_id))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_installed_wheel(name: str, wheel: Path, required_paths: list[Path]) -> str:
    """Bind imported modules and all package members to the supplied artifact."""
    dist = distribution(name)
    root = Path(dist.locate_file(name)).resolve()
    required_members = {f"{name}/{path.resolve().relative_to(root).as_posix()}" for path in required_paths}
    with ZipFile(wheel) as archive:
        members = {
            member for member in archive.namelist() if member.startswith(f"{name}/") and not member.endswith("/")
        }
        if not members or not required_members.issubset(members):
            raise ValueError(f"supplied wheel is missing required {name} module members")
        for member in members:
            installed_path = Path(dist.locate_file(member)).resolve()
            if not installed_path.is_relative_to(root):
                raise ValueError("wheel member is outside the installed package")
            with archive.open(member) as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != _sha256(installed_path):
                    raise ValueError(f"installed {member} differs from the supplied wheel")
    return _sha256(wheel)


def _normalized_cases(directory: Path) -> list[dict[str, object]]:
    """Consume real normalized files with independent native Logistic oracles."""
    import innovate

    covariance = pa.list_(pa.field("covariance_element", pa.float64(), metadata={b"unit": b"AUD^2"}), 2)
    tables = {
        "simple": pa.table({"time": [0.0, 1.0, 2.0], "sample_id": ["a", "b", "c"]}),
        "multi_chunk": pa.table(
            {
                "time": pa.chunked_array([[0.0, 1.0], [2.0]]),
                "sample_id": pa.chunked_array([["a", "b"], ["c"]]),
            }
        ),
        "parquet_normalized_types": pa.table(
            {
                "time": [0.0, 1.0, 2.0],
                "sample_id": ["a", "b", "c"],
                "at": pa.array([0, 1, 2], type=pa.timestamp("s")),
                "covariance": pa.array([[1.0, 0.25], [0.25, 4.0], [2.0, 0.5]], type=covariance),
            }
        ).replace_schema_metadata({b"fixture": b"synthetic-normalized-types"}),
    }
    results = []
    for case, table in tables.items():
        manifest = DatasetManifest(
            dataset_id=f"innovate-{case}",
            tables=(
                TableManifest(
                    table_id="observations",
                    fields=tuple(
                        FieldManifest(
                            field_id=field.name, dtype=str(field.type), unit="day" if field.name == "time" else None
                        )
                        for field in table.schema
                    ),
                ),
            ),
            provenance=SourceProvenance(
                provider_id="innovate-qualification",
                source_uri=f"file:///synthetic/{case}",
                descriptor_digest=hashlib.sha256(case.encode()).hexdigest(),
            ),
        )
        bundle = NormalizedInputBundle(manifest=manifest, tables={"observations": table})
        expected_digest = bundle.content_digest
        for format in ("ipc", "parquet"):
            path = directory / f"{case}.{format}"
            getattr(bundle, f"write_{format}")(path)
            restored = getattr(NormalizedInputBundle, f"read_{format}")(path)
            if not restored.table("observations").schema.equals(table.schema, check_metadata=True):
                raise ValueError("normalized round trip changed Arrow schema metadata")
            if restored.content_digest != expected_digest:
                raise ValueError("normalized round trip changed content_digest")
            frame = read_normalized_frame(path, format=format, table_id="observations", expected_digest=expected_digest)
            # The full source schema/digest is verified above. Timestamp and
            # covariance are provenance-only here, not additional native dtype
            # capability claims. Select the explicit numerical input boundary.
            native_frame = frame.select("time", "sample_id")
            predicted = innovate.predict("logistic", native_frame, {"L": 4.0, "k": log(3.0), "x0": 1.0})
            if not predicted.select(native_frame.columns).equals(native_frame):
                raise ValueError("native normalized consumer changed input rows")
            # These three values follow from exp(log(3)) = 3, independently of
            # the implementation and without generating the expected curve in Innovate.
            values = predicted["prediction"].to_list()
            if not all(
                isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12)
                for actual, expected in zip(values, (1.0, 2.0, 3.0), strict=True)
            ):
                raise ValueError("normalized native Logistic oracle mismatch")
            output = innovate.export_ipc_file(predicted)
            if not pl.read_ipc(io.BytesIO(output)).equals(predicted):
                raise ValueError("normalized consumer output IPC changed predictions")
            results.append(
                {
                    "case": case,
                    "format": format,
                    "input_file_sha256": _sha256(path),
                    "source_content_digest": expected_digest,
                    "restored_content_digest": restored.content_digest,
                    "schema_fingerprint": bundle.schema_fingerprint,
                    "source_chunk_lengths": [[len(chunk) for chunk in column.chunks] for column in table.columns],
                    "restored_chunk_lengths": [
                        [len(chunk) for chunk in column.chunks] for column in restored.table("observations").columns
                    ],
                    "native_input_columns": native_frame.columns,
                    "expected_predictions": [1.0, 2.0, 3.0],
                    "installed_predictions": values,
                    "output_ipc_sha256": hashlib.sha256(output).hexdigest(),
                }
            )
    return results


def qualify(wheel: Path | None = None, voiage_wheel: Path | None = None) -> dict[str, object]:
    from voiage import _core

    import innovate
    from innovate import _native

    if sys.version_info[:3] != (3, 15, 2):
        raise ValueError("qualification requires repo-local Python 3.15.2")
    checkout = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve().is_relative_to(checkout):
        raise ValueError("installed qualification must run outside the checkout")
    for prohibited in ("pandas", "xarray"):
        if find_spec(prohibited) is not None:
            raise ValueError(f"installed consumer unexpectedly includes {prohibited}")
    record_hashes = {}
    imported_paths = {}
    artifact = {}
    for name, module in (("innovate", innovate), ("voiage", voiage)):
        dist = distribution(name)
        init = Path(dist.locate_file(f"{name}/__init__.py")).resolve()
        if not dist.read_text("WHEEL") or not init.is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError(f"{name} is not an installed wheel")
        if init != Path(module.__file__).resolve():
            raise ValueError(f"{name} import origin is shadowed")
        if init.is_relative_to(checkout):
            raise ValueError(f"{name} is installed inside the source checkout")
        imported_paths[name] = str(init)
        record = dist.read_text("RECORD")
        if not record:
            raise ValueError(f"{name} wheel has no installed RECORD")
        record_hashes[name] = hashlib.sha256(record.encode()).hexdigest()
        supplied_wheel = wheel if name == "innovate" else voiage_wheel
        if supplied_wheel is not None:
            required = (
                [init, Path(_native.__file__)]
                if name == "innovate"
                else [init, Path(normalized_input.__file__), Path(_core.__file__)]
            )
            artifact[f"{name}_wheel_sha256"] = verify_installed_wheel(name, supplied_wheel, required)
    native_path = Path(_native.__file__).resolve()
    if native_path.parent != Path(innovate.__file__).resolve().parent:
        raise ValueError("Innovate native extension import origin is shadowed")
    with native_path.open("rb") as native_binary:
        native_sha256 = hashlib.file_digest(native_binary, "sha256").hexdigest()
    imported_paths["innovate._native"] = str(native_path)
    contract_path = Path(normalized_input.__file__).resolve()
    if not contract_path.is_relative_to(Path(voiage.__file__).resolve().parent):
        raise ValueError("Voiage normalized contract import origin is shadowed")
    imported_paths["voiage.contracts.normalized_input"] = str(contract_path)
    voiage_native_path = Path(_core.__file__).resolve()
    if voiage_native_path.parent != Path(voiage.__file__).resolve().parent:
        raise ValueError("Voiage native extension import origin is shadowed")
    imported_paths["voiage._core"] = str(voiage_native_path)
    imported_paths["polars"] = str(Path(pl.__file__).resolve())
    imported_paths["pyarrow"] = str(Path(pa.__file__).resolve())
    with TemporaryDirectory(prefix="innovate-normalized-") as temporary:
        normalized_cases = _normalized_cases(Path(temporary))
    rates = (0.01, 0.08)
    scenarios = [
        {
            "id": f"diffusion-{index}",
            "model_key": "bass",
            "simulation": {"parameters": {"p": p, "q": 0.3, "m": 100.0}},
        }
        for index, p in enumerate(rates)
    ]
    scenarios.append(
        {
            "id": "alternative",
            "model_key": "logistic",
            "simulation": {"parameters": {"L": 32.0, "k": 1.0, "x0": 3.0}},
        }
    )
    response = innovate.execute_scenario_grid(pl.DataFrame({"time": [0.0, 3.0]}), scenarios=scenarios)
    executions = response["result"]["executions"]
    if [row["id"] for row in executions] != [scenario["id"] for scenario in scenarios]:
        raise ValueError("native scenario order changed")
    if any(row["error"] is not None for row in executions):
        raise ValueError("native scenario execution failed")
    values = [row["result"]["values"][-1] for row in executions[:2]]
    for p, value in zip(rates, values, strict=True):
        decay = exp(-(p + 0.3) * 3.0)
        expected = 100.0 * (1.0 - decay) / (1.0 + 0.3 / p * decay)
        if not isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("native Bass analytic oracle mismatch")
    alternative = executions[2]["result"]["values"][-1]
    if alternative != 16.0:
        raise ValueError("native Logistic midpoint oracle mismatch")
    payoffs = pl.DataFrame({"diffusion": values, "alternative": [alternative] * len(values)})
    data = innovate.export_ipc_file(payoffs)
    restored = pl.read_ipc(io.BytesIO(data))
    if not restored.equals(payoffs):
        raise ValueError("Arrow IPC changed scenario payoff data")
    expected = (16.0 - values[0]) / 2.0
    observed = float(evpi(ValueArray.from_polars(restored)))
    if not (values[0] < 16 < values[1] and sum(values) / 2 > 16):
        raise ValueError("fixture decisions must switch across scenarios")
    if not isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Voiage EVPI oracle mismatch")
    # Four equally likely draws encode a 3:1 empirical scenario distribution.
    # Voiage 3.1.0 EVPI consumes draws without a separate weight argument.
    weighted = restored[[0, 0, 0, 1]]
    weighted_data = innovate.export_ipc_file(weighted)
    weighted_restored = pl.read_ipc(io.BytesIO(weighted_data))
    if not weighted_restored.equals(weighted):
        raise ValueError("Arrow IPC changed empirical scenario multiplicities")
    if (3 * values[0] + values[1]) / 4 >= alternative:
        raise ValueError("weighted fixture must prefer the alternative")
    weighted_expected = (values[1] - alternative) / 4
    weighted_observed = float(evpi(ValueArray.from_polars(weighted_restored)))
    if not isclose(weighted_observed, weighted_expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Voiage empirical-weight EVPI oracle mismatch")
    return {
        **artifact,
        "python": sys.version.split()[0],
        "innovate": version("innovate"),
        "voiage": version("voiage"),
        "polars": pl.__version__,
        "pyarrow": pa.__version__,
        "imported_paths": imported_paths,
        "voiage_normalized_module_sha256": _sha256(contract_path),
        "voiage_native_sha256": _sha256(Path(_core.__file__)),
        "installed_record_sha256": record_hashes,
        "innovate_native_sha256": native_sha256,
        "arrow_ipc_sha256": hashlib.sha256(data).hexdigest(),
        "reference_evpi": expected,
        "installed_evpi": observed,
        "weighted_arrow_ipc_sha256": hashlib.sha256(weighted_data).hexdigest(),
        "weighted_reference_evpi": weighted_expected,
        "weighted_installed_evpi": weighted_observed,
        "normalized_cases": normalized_cases,
        "scope": "Synthetic CPU scenario handoff only; not full scientific/HPC/clinical acceptance",
    }


def main() -> int:
    """Run the installed consumer in an isolated process outside the checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True, help="Innovate wheel used for the installed consumer")
    parser.add_argument("--voiage-wheel", type=Path, help="Optional Voiage wheel to compare with the installed package")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not sys.flags.isolated:
            raise ValueError("qualification worker requires Python isolated mode")
        print(json.dumps(qualify(args.wheel, args.voiage_wheel), indent=2))
        return 0
    script = Path(__file__).resolve()
    command = [sys.executable, "-I", str(script), "--worker", "--wheel", str(args.wheel.resolve())]
    if args.voiage_wheel is not None:
        command += ["--voiage-wheel", str(args.voiage_wheel.resolve())]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=script.parents[1], check=True, capture_output=True, text=True
    ).stdout.strip()
    with TemporaryDirectory(prefix="innovate-installed-consumer-") as temporary:
        process = subprocess.run(command, cwd=temporary, capture_output=True, text=True, check=False)
        if process.stderr:
            print(process.stderr, file=sys.stderr, end="")
        receipt = json.loads(process.stdout) if process.returncode == 0 else {}
        receipt.update(
            status="passed" if process.returncode == 0 else "failed",
            command=command,
            exit_status=process.returncode,
            working_directory=temporary,
            source_checkout=str(script.parents[1]),
            qualification_source_revision=revision,
            qualification_script_sha256=_sha256(script),
        )
    print(json.dumps(receipt, indent=2))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
