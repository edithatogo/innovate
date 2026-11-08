"""Measure Bass capacity and mean-trajectory interval coverage under one fixed-grid noise design."""

import json
import math
import random
import statistics

import innovate

TRIALS = 64
DRAWS = 100
CAPACITY = 120.0
TIMES = list(range(1, 25))
FORECAST_TIMES = [25, 30, 40]


def truth(time):
    decay = math.exp(-0.24 * time)
    return CAPACITY * (1 - decay) / (1 + 7 * decay)


def main():
    rng = random.Random(2431)
    cases = []
    for _ in range(TRIALS):
        observed = []
        for t in TIMES:
            observed.append(truth(t) + rng.gauss(0, 0.5))
        cases.append({"inputs": {"time": TIMES, "observed": observed}})

    for method in ("paired_case_resampling", "centered_residual_resampling"):
        widths = []
        covered = failed_refits = 0
        refusals = []
        forecasts = {t: {"covered": 0, "widths": []} for t in FORECAST_TIMES}
        forecast_failures = 0
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
                    forecast_inputs={"time": FORECAST_TIMES},
                )["result"]
            except innovate.NativeError as error:
                refusals.append({"trial": trial, "error": str(error)})
                continue
            failed_refits += result["failed"]
            forecast = result["forecast"]
            forecast_failures += forecast["failed"]
            assert forecast["time"] == FORECAST_TIMES
            for t, row in zip(FORECAST_TIMES, forecast["pointwise_intervals"], strict=True):
                if row[0] is not None:
                    lower, upper = row[0]
                    forecasts[t]["widths"].append(upper - lower)
                    forecasts[t]["covered"] += lower <= truth(t) <= upper
            bounds = result["parameter_intervals"]["m"]
            if bounds is not None:
                lower, upper = bounds
                widths.append(upper - lower)
                covered += lower <= CAPACITY <= upper
        for stats in forecasts.values():
            samples = stats.pop("widths")
            stats["returned_intervals"] = len(samples)
            stats["mean_width"] = statistics.mean(samples) if samples else None
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
                    "forecast_intervals": forecasts,
                    "failed_forecasts": forecast_failures,
                }
            )
        )


if __name__ == "__main__":
    main()
