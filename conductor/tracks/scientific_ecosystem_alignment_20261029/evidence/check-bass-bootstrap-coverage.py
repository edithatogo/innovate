"""Measure Bass capacity interval coverage under one fixed-grid noise design."""

import json
import math
import random
import statistics

import innovate

TRIALS = 64
DRAWS = 100
CAPACITY = 120.0
TIMES = list(range(1, 25))


def main():
    rng = random.Random(2431)
    cases = []
    for _ in range(TRIALS):
        observed = []
        for t in TIMES:
            decay = math.exp(-0.24 * t)
            observed.append(CAPACITY * (1 - decay) / (1 + 7 * decay) + rng.gauss(0, 0.5))
        cases.append({"inputs": {"time": TIMES, "observed": observed}})

    for method in ("paired_case_resampling", "centered_residual_resampling"):
        widths = []
        covered = failed_refits = 0
        refusals = []
        for trial, fit in enumerate(cases):
            try:
                result = innovate.execute(
                    "bootstrap_fit",
                    fit_model="bass",
                    fit=fit,
                    draws=DRAWS,
                    interval_mass=0.9,
                    seed=3000 + trial,
                    method=method,
                )["result"]
            except innovate.NativeError as error:
                refusals.append({"trial": trial, "error": str(error)})
                continue
            failed_refits += result["failed"]
            bounds = result["parameter_intervals"]["m"]
            if bounds is not None:
                lower, upper = bounds
                widths.append(upper - lower)
                covered += lower <= CAPACITY <= upper
        print(
            json.dumps(
                {
                    "method": method,
                    "trials": TRIALS,
                    "returned_intervals": len(widths),
                    "covered": covered,
                    "mean_width": statistics.mean(widths) if widths else None,
                    "failed_refits": failed_refits,
                    "attempted_refits": (TRIALS - len(refusals)) * DRAWS,
                    "refusals": refusals,
                }
            )
        )


if __name__ == "__main__":
    main()
