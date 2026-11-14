"""Verify ordered batch results, analytic values and worker-count validation."""

import json
from math import exp, isclose

from innovate import NativeError, predict_batch


def main() -> None:
    """Compare one and two workers using two distinct synthetic Logistic curves."""
    datasets = [
        {"inputs": {"time": [0, 1]}, "parameters": {"L": 100, "k": 1, "x0": 0}},
        {"inputs": {"time": [0]}, "parameters": {"L": 200, "k": 1, "x0": 0}},
    ]
    rows = predict_batch("logistic", datasets, workers=2)
    assert rows == predict_batch("logistic", datasets, workers=1)
    assert len(rows) == 2 and all(row["error"] is None for row in rows)
    assert rows[0]["result"]["values"][0] == 50
    assert isclose(rows[0]["result"]["values"][1], 100 / (1 + exp(-1)), abs_tol=1e-12, rel_tol=0)
    assert rows[1]["result"]["values"] == [100]
    rejected = False
    try:
        predict_batch("logistic", datasets, workers=0)
    except NativeError as error:
        rejected = error.code == "invalid_request"
    assert rejected
    print(
        json.dumps(
            {"batch_items": len(rows), "invalid_workers_rejected": rejected, "worker_invariant": True}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
