"""Check native Logistic prediction against an independent analytic reference.

Synthetic values have arbitrary adoption units; no empirical fit or calibrated
forecast uncertainty is implied. Run with the installed development package.
"""

import json
import math

import polars as pl

from innovate import predict


def main() -> None:
    """Validate predictions, preserved identity, LazyFrames and invalid grids."""
    times = [0.0, 1.0, 2.0]
    parameters = {"L": 100.0, "k": 1.0, "x0": 1.0}
    frame = pl.DataFrame({"time": times, "id": ["a", "b", "c"]})
    forecast = predict("logistic", frame, parameters)

    # This independent closed form is a test oracle, not a production fallback.
    expected = [100.0 / (1.0 + math.exp(-(time - 1.0))) for time in times]
    actual = forecast["prediction"].to_list()
    assert all(
        math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-12)
        for value, reference in zip(actual, expected, strict=True)
    )
    assert forecast["id"].to_list() == ["a", "b", "c"]
    assert forecast["time"].to_list() == times
    assert frame.columns == ["time", "id"]
    assert predict("logistic", frame.lazy(), parameters).equals(forecast)

    try:
        predict("logistic", pl.DataFrame({"time": [0.0, 0.0]}), parameters)
    except ValueError:
        duplicate_time_rejected = True
    else:
        duplicate_time_rejected = False
    assert duplicate_time_rejected, "The native contract must reject duplicate times."

    print(
        json.dumps(
            {
                "duplicate_time_rejected": duplicate_time_rejected,
                "ids": forecast["id"].to_list(),
                "predictions": [round(value, 6) for value in actual],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
