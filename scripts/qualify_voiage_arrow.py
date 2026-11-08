"""Installed native Innovate → Polars/Arrow → Voiage consumer qualification."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from importlib.metadata import distribution, version
from math import exp, isclose
from pathlib import Path
from zipfile import ZipFile

import polars as pl
import voiage
from voiage.methods.basic import evpi
from voiage.schema import ValueArray

import innovate
from innovate import _native


def qualify(wheel: Path | None = None) -> dict[str, object]:
    if sys.version_info[:3] != (3, 15, 2):
        raise ValueError("qualification requires repo-local Python 3.15.2")
    record_hashes = {}
    for name, module in (("innovate", innovate), ("voiage", voiage)):
        dist = distribution(name)
        init = Path(dist.locate_file(f"{name}/__init__.py")).resolve()
        if not dist.read_text("WHEEL") or not init.is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError(f"{name} is not an installed wheel")
        if init != Path(module.__file__).resolve():
            raise ValueError(f"{name} import origin is shadowed")
        record = dist.read_text("RECORD")
        if not record:
            raise ValueError(f"{name} wheel has no installed RECORD")
        record_hashes[name] = hashlib.sha256(record.encode()).hexdigest()
    native_path = Path(_native.__file__).resolve()
    if native_path.parent != Path(innovate.__file__).resolve().parent:
        raise ValueError("Innovate native extension import origin is shadowed")
    with native_path.open("rb") as native_binary:
        native_sha256 = hashlib.file_digest(native_binary, "sha256").hexdigest()
    artifact = {}
    if wheel is not None:
        with ZipFile(wheel) as archive:
            for module_path in (Path(innovate.__file__), native_path):
                with archive.open(f"innovate/{module_path.name}") as member, module_path.open("rb") as installed:
                    member_digest = hashlib.file_digest(member, "sha256").digest()
                    if member_digest != hashlib.file_digest(installed, "sha256").digest():
                        raise ValueError(f"installed {module_path.name} differs from the supplied wheel")
        with wheel.open("rb") as package:
            artifact["innovate_wheel_sha256"] = hashlib.file_digest(package, "sha256").hexdigest()
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
        "installed_record_sha256": record_hashes,
        "innovate_native_sha256": native_sha256,
        "arrow_ipc_sha256": hashlib.sha256(data).hexdigest(),
        "reference_evpi": expected,
        "installed_evpi": observed,
        "weighted_arrow_ipc_sha256": hashlib.sha256(weighted_data).hexdigest(),
        "weighted_reference_evpi": weighted_expected,
        "weighted_installed_evpi": weighted_observed,
        "scope": "Synthetic CPU scenario handoff only; not full scientific/HPC/clinical acceptance",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="Innovate wheel used for the installed consumer")
    print(json.dumps(qualify(parser.parse_args().wheel), indent=2))
