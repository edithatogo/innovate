"""Reproduce the exploratory one-decade US population interval comparison."""

import json
import subprocess

import polars as pl

import innovate


def main() -> None:
    raw = subprocess.check_output(
        ["Rscript", "--vanilla", "-e", "cat(as.numeric(datasets::uspop), sep=\",\")"],
        text=True,
    )
    observed = [float(value) for value in raw.split(",")]
    time = list(range(19))
    if len(observed) != len(time):
        raise ValueError("expected 19 decennial observations from 1790 to 1970")

    results = {}
    for model in ("logistic", "gompertz", "bass"):
        predicted = []
        for origin in range(9, len(time)):
            fit = innovate.fit(model, pl.DataFrame({"time": time[:origin], "observed": observed[:origin]}))
            predicted.append(fit.predict(pl.DataFrame({"time": [origin]}))["prediction"][0])
        result = innovate.execute(
            "calibrate_prediction_intervals",
            method="split_conformal",
            time=time[9:],
            observed=observed[9:],
            predicted=predicted,
            confidence=0.8,
            holdout=[0] * 5 + [1] * 5,
        )["result"]
        results[model] = {
            "half_width_millions": result["metadata"]["interval_half_width"],
            "evaluation_coverage": result["diagnostics"]["holdout_coverage"],
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
