"""Installed native Innovate → Polars/Arrow → Voiage consumer qualification."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from importlib.metadata import distribution, version
from math import exp, isclose
from pathlib import Path

import polars as pl
import voiage
from voiage.methods.basic import evpi
from voiage.schema import ValueArray

import innovate


def qualify() -> dict[str, object]:
    if sys.version_info[:3] != (3, 15, 2):
        raise ValueError("qualification requires repo-local Python 3.15.2")
    for name, module in (("innovate", innovate), ("voiage", voiage)):
        dist = distribution(name)
        init = Path(dist.locate_file(f"{name}/__init__.py")).resolve()
        if not dist.read_text("WHEEL") or not init.is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError(f"{name} is not an installed wheel")
        if init != Path(module.__file__).resolve():
            raise ValueError(f"{name} import origin is shadowed")
    values = []
    for p in (0.01, 0.08):
        result = innovate.predict("bass", pl.DataFrame({"time": [0.0, 3.0]}), {"p": p, "q": 0.3, "m": 100.0})
        value = float(result["prediction"][-1])
        decay = exp(-(p + 0.3) * 3.0)
        expected = 100.0 * (1.0 - decay) / (1.0 + 0.3 / p * decay)
        if not isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("native Bass analytic oracle mismatch")
        values.append(value)
    payoffs = pl.DataFrame({"diffusion": values, "alternative": [16.0, 16.0]})
    data = payoffs.write_ipc(None).getvalue()
    restored = pl.read_ipc(io.BytesIO(data))
    if not restored.equals(payoffs):
        raise ValueError("Arrow IPC changed scenario payoff data")
    expected = (16.0 - values[0]) / 2.0
    observed = float(evpi(ValueArray.from_polars(restored)))
    if not (values[0] < 16 < values[1] and sum(values) / 2 > 16):
        raise ValueError("fixture decisions must switch across scenarios")
    if not isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Voiage EVPI oracle mismatch")
    return {
        "python": sys.version.split()[0],
        "innovate": version("innovate"),
        "voiage": version("voiage"),
        "polars": pl.__version__,
        "arrow_ipc_sha256": hashlib.sha256(data).hexdigest(),
        "reference_evpi": expected,
        "installed_evpi": observed,
        "scope": "Synthetic CPU scenario handoff only; not full scientific/HPC/clinical acceptance",
    }


if __name__ == "__main__":
    print(json.dumps(qualify(), indent=2))
