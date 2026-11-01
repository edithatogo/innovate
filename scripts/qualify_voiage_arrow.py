"""Exercise installed Innovate and Voiage wheels through an Arrow IPC handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import distribution, version
from math import exp, isclose
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import voiage
from voiage.methods.basic import evpi

import innovate
from innovate.arrow_interchange import kernel_array_payload_from_table, kernel_array_payload_to_table
from innovate.diffuse.bass import BassModel
from innovate.kernel import KernelArrayPayload


def qualify(*, index_install: bool = False) -> dict[str, object]:
    """Check package provenance, Arrow round-trip and finite-state decision value."""
    if sys.version_info[:3] != (3, 15, 1) or pa.__version__.split(".")[0] != "26":
        raise ValueError("This qualification requires Python 3.15.1 and PyArrow 26")
    for name, module in (("innovate", innovate), ("voiage", voiage)):
        dist = distribution(name)
        origin = json.loads(dist.read_text("direct_url.json") or "{}")
        if index_install:
            if origin or not dist.read_text("WHEEL"):
                raise ValueError(f"{name} must be installed from the index in wheel format")
        elif "archive_info" not in origin or not origin.get("url", "").endswith(".whl"):
            raise ValueError(f"{name} must be installed from a wheel")
        init = Path(dist.locate_file(f"{name}/__init__.py")).resolve()
        if not init.is_relative_to(Path(sys.prefix).resolve()) or init != Path(module.__file__).resolve():
            raise ValueError(f"Unexpected {name} import origin")

    # Two equally likely diffusion scenarios; the alternative earns a fixed 16 units.
    values = []
    reference = []
    for p in (0.01, 0.08):
        model = BassModel()
        model.params_ = {"p": p, "q": 0.3, "m": 100.0}
        values.extend((float(model.predict([0.0, 3.0])[-1]), 16.0))
        decay = exp(-(p + 0.3) * 3.0)
        reference.append(100.0 * (1.0 - decay) / (1.0 + 0.3 / p * decay))
    # The NumPy backend integrates the diffusion ODE numerically.
    np.testing.assert_allclose(values[::2], reference, rtol=1e-7, atol=1e-8)
    payload = KernelArrayPayload.from_values(values, (2, 2), "float64", {"unit": "synthetic payoff"})
    table = kernel_array_payload_to_table(payload)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "handoff.arrow"
        with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        with pa.memory_map(str(path), "r") as source:
            restored = kernel_array_payload_from_table(pa.ipc.open_file(source).read_all())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if restored != payload:
        raise ValueError("Arrow round-trip changed the kernel payload")
    # A fixed action chooses diffusion on average. Perfect information changes
    # the decision only in the lower-adoption scenario.
    if not (values[0] < 16 < values[2] and (values[0] + values[2]) / 2 > 16):
        raise ValueError("Expected the decisions to switch across scenarios")
    expected = (16 - values[0]) / 2
    observed = float(evpi(np.asarray(restored.values).reshape(restored.shape)))
    if expected <= 0 or not isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Voiage EVPI {observed} differs from the independent reference {expected}")
    return {
        "python": sys.version.split()[0],
        "innovate": version("innovate"),
        "voiage": version("voiage"),
        "pyarrow": pa.__version__,
        "arrow_ipc_sha256": digest,
        "reference_evpi": expected,
        "installed_evpi": observed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-install", action="store_true", help="Check installations resolved from a package index"
    )
    print(json.dumps(qualify(index_install=parser.parse_args().index_install), indent=2))
