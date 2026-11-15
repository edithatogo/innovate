#!/usr/bin/env python3
"""Reproduce independent normal-distribution references; never import Innovate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp


DIRECTORY = Path(__file__).resolve().parent
MPMATH_VERSION = "1.3.0"
PRECISION = 400
BASE_SHA = "42189a223fd44c823ca11aec689ea67b11c25790"


def encoded(value: mp.mpf | float) -> float | str:
    """Round once to binary64 and represent nonfinite results as JSON strings."""
    result = float(value)
    if math.isnan(result):
        return "nan"
    if math.isinf(result):
        return "inf" if result > 0 else "-inf"
    return result


def exact(value: float | str) -> mp.mpf:
    """Use the supplied binary64 value, not its rounded decimal spelling."""
    return mp.mpf(float(value))


def reference(value: mp.mpf) -> dict[str, object]:
    """Retain a readable high-precision value as well as its binary64 rounding."""
    return {"expected": encoded(value), "high_precision": mp.nstr(value, 65)}


def make_fixture() -> dict[str, object]:
    """Evaluate analytic definitions without Rust, libm, SciPy, or statrs."""
    if mp.__version__ != MPMATH_VERSION:
        raise RuntimeError(f"expected mpmath {MPMATH_VERSION}, got {mp.__version__}")
    mp.mp.dps = PRECISION
    sqrt_two = mp.sqrt(2)
    log_normalizer = mp.log(2 * mp.pi) / 2

    probabilities = [
        ("endpoint_zero", 0.0),
        ("far_lower_tail", 1e-300),
        ("outside_legacy_bound", 1e-20),
        ("lower_tail_1e12", 1e-12),
        ("lower_tail_1e8", 1e-8),
        ("required_lower_tail", 1e-6),
        ("smallest_reachable_blom", 0.625 / 160000.25),
        ("lower_1e4", 1e-4),
        ("lower_1e3", 1e-3),
        ("required_lower_025", 0.025),
        ("lower_decile", 0.1),
        ("lower_quartile", 0.25),
        ("required_median", 0.5),
        ("upper_quartile", 0.75),
        ("upper_decile", 0.9),
        ("required_upper_975", 0.975),
        ("largest_reachable_blom", (160000 - 0.375) / 160000.25),
        ("required_upper_tail", 1 - 1e-6),
        ("upper_tail_1e8", 1 - 1e-8),
        ("upper_tail_1e12", 1 - 1e-12),
        ("largest_probability_below_one", math.nextafter(1.0, 0.0)),
        ("endpoint_one", 1.0),
    ]
    quantiles = []
    for name, probability in probabilities:
        value = sqrt_two * mp.erfinv(2 * exact(probability) - 1)
        quantiles.append(
            {
                "id": name,
                "p": probability,
                "p_hex": probability.hex(),
                **reference(value),
                "abs_tol": 2e-10,
                "legacy_domain": 1e-6 <= probability <= 1 - 1e-6,
                "scope": "endpoint" if probability in (0, 1) else "distribution",
            }
        )

    cdf = []
    for name, value in [
        ("negative_infinity", "-inf"),
        ("lower_underflow", -40.0),
        ("lower_tail_10", -10.0),
        ("lower_tail_8", -8.0),
        ("lower_tail_5", -5.0),
        ("lower_2", -2.0),
        ("lower_1", -1.0),
        ("lower_near_center", -1e-10),
        ("center", 0.0),
        ("upper_near_center", 1e-10),
        ("upper_1", 1.0),
        ("upper_2", 2.0),
        ("upper_tail_5", 5.0),
        ("upper_tail_8", 8.0),
        ("upper_tail_10", 10.0),
        ("upper_underflow", 40.0),
        ("positive_infinity", "inf"),
        ("nan", "nan"),
    ]:
        x = exact(value)
        lower = mp.erfc(-x / sqrt_two) / 2
        upper = mp.erfc(x / sqrt_two) / 2
        cdf.append(
            {
                "id": name,
                "x": value,
                **reference(lower),
                "survival": encoded(upper),
                "survival_high_precision": mp.nstr(upper, 65),
                "abs_tol": 0.0,
                "rel_tol": 0.0 if value == 0.0 else 2e-13,
            }
        )

    density_inputs = [
        ("standard_center", 0.0, 0.0, 1.0),
        ("near_center_1e10", 1e-10, 0.0, 1.0),
        ("near_center_1e8", 1e-8, 0.0, 1.0),
        ("standard_negative", -1.0, 0.0, 1.0),
        ("standard_positive", 1.0, 0.0, 1.0),
        ("large_negative", -40.0, 0.0, 1.0),
        ("large_positive", 40.0, 0.0, 1.0),
        ("large_finite_log_density", 1e154, 0.0, 1.0),
        ("overflow_log_density", 1e200, 0.0, 1.0),
        ("shifted_scaled", 3.5, 2.0, 0.5),
        ("shifted_at_mean", 2.0, 2.0, 0.5),
        ("tiny_scale_unit_residual", 1e-300, 0.0, 1e-300),
        ("huge_scale_unit_residual", 1e300, 0.0, 1e300),
        ("tiny_scale_at_mean", 0.0, 0.0, 1e-300),
        ("huge_scale_at_mean", 0.0, 0.0, 1e300),
        ("negative_infinity", "-inf", 0.0, 1.0),
        ("positive_infinity", "inf", 0.0, 1.0),
        ("nan", "nan", 0.0, 1.0),
    ]
    densities = []
    relative_densities = []
    for name, value, mean, sd in density_inputs:
        z = (exact(value) - exact(mean)) / exact(sd)
        relative = -(z * z) / 2
        density = relative - mp.log(exact(sd)) - log_normalizer
        inputs = {"id": name, "x": value, "mean": mean, "sd": sd}
        densities.append(
            {
                **inputs,
                **reference(density),
                "abs_tol": 2e-13,
                "rel_tol": 5e-15,
            }
        )
        relative_densities.append(
            {
                **inputs,
                **reference(relative),
                "abs_tol": 0.0,
                "rel_tol": 5e-15,
            }
        )

    # The endpoint of a 48-step binary partition of [-8, 8] is analytic.
    # These values characterize unreachable private inputs; they are not
    # mathematical normal quantiles and must not become a public contract.
    edge_offset = 2.0**-45
    legacy_boundaries = [
        {"id": name, "p": value, "expected": expected, "abs_tol": 0.0}
        for name, value, expected in [
            ("zero", 0.0, -8.0 + edge_offset),
            ("below_zero", -1.0, -8.0 + edge_offset),
            ("negative_infinity", "-inf", -8.0 + edge_offset),
            ("nan", "nan", -8.0 + edge_offset),
            ("one", 1.0, 8.0 - edge_offset),
            ("above_one", 2.0, 8.0 - edge_offset),
            ("positive_infinity", "inf", 8.0 - edge_offset),
        ]
    ]

    input_groups = {
        "quantile": probabilities,
        "cdf": [(row["id"], row["x"]) for row in cdf],
        "density": density_inputs,
        "legacy_boundaries": [(row["id"], row["p"]) for row in legacy_boundaries],
    }
    input_bytes = json.dumps(
        input_groups, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "schema_version": 1,
        "qualification": "synthetic mathematical references; no scientific-release claim",
        "provenance": {
            "base_commit": BASE_SHA,
            "oracle_package": "mpmath",
            "oracle_version": MPMATH_VERSION,
            "decimal_precision": PRECISION,
            "input_conversion": "mp.mpf(binary64_input); exact represented input, not decimal ideal",
            "expected_rounding": "one final conversion of high-precision result to binary64",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "generator_command": "python generate.py --check",
            "sources": [
                "https://mpmath.org/doc/1.3.0/functions/expintegrals.html#erfinv",
                "https://mpmath.org/doc/1.3.0/basics.html#number-types",
            ],
        },
        "normal_quantile": quantiles,
        "normal_cdf": cdf,
        "normal_log_pdf": densities,
        "gaussian_relative_log_density": relative_densities,
        "legacy_quantile_boundaries": legacy_boundaries,
    }


def main() -> int:
    """Write or byte-check the sole sibling fixture; print its content hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if fixture bytes differ")
    args = parser.parse_args()
    output = json.dumps(make_fixture(), indent=2, allow_nan=False) + "\n"
    target = DIRECTORY / "normal.json"
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != output:
            print("FAIL: normal.json differs from the independent generator")
            return 1
    else:
        target.write_text(output, encoding="utf-8")
    print(f"PASS: normal.json sha256={hashlib.sha256(output.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
