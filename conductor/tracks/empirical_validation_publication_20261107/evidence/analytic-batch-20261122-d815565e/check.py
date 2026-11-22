import json
import math
import sys
from decimal import Decimal, localcontext
from hashlib import sha256
from itertools import product
from pathlib import Path
from platform import machine
from zipfile import ZipFile

import polars as pl

import innovate
from innovate import _native

assert sys.version_info[:3] == (3, 15, 2)
assert sys.platform == "darwin" and machine() == "arm64"
assert Path(innovate.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert (
    not Path.cwd().resolve().is_relative_to(Path("/Users/doughnut/.codex/worktrees/innovate-pr671-ci-repair/innovate"))
)
(wheel,) = Path.cwd().glob("*.whl")
wheel_sha256 = sha256(wheel.read_bytes()).hexdigest()
assert wheel_sha256 == "5de562362fad24ea88674d6b6e425bd0304a0c3d9faf5c724862869ba602c72b"
native_library = Path(_native.__file__).resolve()
assert native_library.is_relative_to(Path(sys.prefix).resolve())
with ZipFile(wheel) as archive:
    assert archive.read("innovate/__init__.py") == Path(innovate.__file__).read_bytes()
    assert archive.read(f"innovate/{native_library.name}") == native_library.read_bytes()

T = (0.0, 0.25, 1.0, 3.0, 6.0, 12.0)


def value(d):
    return Decimal.from_float(float(d))


def expected(model, p, t):
    with localcontext() as context:
        context.prec = 70
        t = value(t)
        if model == "bass":
            rate, imitation, market = map(value, (p["p"], p["q"], p["m"]))
            e = (-(rate + imitation) * t).exp()
            return float(market * (1 - e) / (1 + imitation * e / rate))
        if model == "logistic":
            level, rate, midpoint = map(value, (p["L"], p["k"], p["x0"]))
            return float(level / (1 + (-rate * (t - midpoint)).exp()))
        if model == "gompertz":
            level, displacement, rate = map(value, (p["a"], p["b"], p["c"]))
            return float(level * (-displacement * (-rate * t).exp()).exp())
        slope, midpoint = map(value, (p["alpha"], p["t0"]))
        return float(1 / (1 + (-slope * (t - midpoint)).exp()))


cases = {
    "bass": [
        dict(zip(("p", "q", "m"), values, strict=True))
        for values in product((0.005, 0.04, 0.25), (0.0, 0.5, 1.8), (1.0, 100.0, 10000.0))
    ],
    "logistic": [
        dict(zip(("L", "k", "x0"), values, strict=True))
        for values in product((1.0, 100.0, 10000.0), (0.005, 0.4, 1.8), (0.0, 3.0, 12.0))
    ],
    "gompertz": [
        dict(zip(("a", "b", "c"), values, strict=True))
        for values in product((1.0, 100.0, 10000.0), (0.1, 1.0, 4.0), (0.005, 0.4, 1.8))
    ],
    "fisher_pry": [
        dict(zip(("alpha", "t0"), values, strict=True)) for values in product((0.005, 0.4, 1.8), (0.0, 3.0, 12.0))
    ],
}
source = pl.DataFrame({"time": T, "row_id": list(range(len(T))), "note": ["a", "b", "c", "d", "e", "f"]})
summary = {}
for model, model_cases in cases.items():
    comparisons = 0
    max_absolute_error = 0.0
    for params in model_cases:
        predicted = innovate.predict(model, source, params)
        assert predicted.select(source.columns).equals(source)
        assert predicted.columns == source.columns + ["prediction"]
        for t, observed in zip(T, predicted["prediction"].to_list(), strict=True):
            target = expected(model, params, t)
            assert math.isclose(observed, target, rel_tol=1e-12, abs_tol=1e-10), (model, params, t, observed, target)
            max_absolute_error = max(max_absolute_error, abs(observed - target))
            comparisons += 1
    summary[model] = {
        "series": len(model_cases),
        "reference_values": comparisons,
        "max_absolute_error": max_absolute_error,
    }

invalid = 0
for model, params in (
    ("bass", {"p": -1.0, "q": 0.5, "m": 1.0}),
    ("logistic", {"L": 1.0, "k": -1.0, "x0": 0.0}),
    ("gompertz", {"a": -1.0, "b": 1.0, "c": 1.0}),
    ("fisher_pry", {"alpha": -1.0, "t0": 0.0}),
):
    try:
        innovate.predict(model, source, params)
    except ValueError, RuntimeError:
        invalid += 1
    else:
        raise AssertionError(f"{model} unexpectedly accepted invalid parameters")
print(
    json.dumps(
        {
            "kind": "bounded_analytic_reference_batch",
            "schema_version": 1,
            "status": "passed",
            "python": "3.15.2",
            "platform": "macos-arm64",
            "wheel_sha256": wheel_sha256,
            "source_commit": "d815565ee740754694ff74468750f27202bafdf2",
            "source_sha256": "188a0e3640509cfc8b61f0bc360ef76418d3be918ecc473e2c06af2e08e5765c",
            "independent_reference": "Decimal exponential with 70 digits and binary64 input conversion",
            "dataframe": "preserved time, row_id, note, order and prediction column",
            "results": summary,
            "invalid_parameter_rejections": invalid,
        },
        indent=2,
    )
)
