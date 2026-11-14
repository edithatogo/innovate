"""Check a seeded Bass DES path, replay and its count-to-level interpretation.

The population has 100 agents, while market potential is 250 arbitrary units.
One seeded path is a synthetic realization, not an expected adoption curve.
Finite-population expectations with imitation need not equal deterministic Bass.
"""

import json
import math
from itertools import pairwise

import polars as pl

from innovate import NativeError, execute


def main() -> None:
    """Validate replay, sampling invariance, physical bounds and engine errors."""
    times = [0.0, 1.0, 5.0, 20.0]
    parameters = {"p": 0.2, "q": 0.3, "m": 250.0}
    options = {"engine": "kairos_des", "seed": 42, "population": 100, "ticks_per_unit": 1_000_000}
    request = {"inputs": {"time": times}, "parameters": parameters, "simulation_options": options}
    response = execute("simulate_model", "bass", **request)
    replay_equal = response == execute("simulate_model", "bass", **request)
    assert replay_equal
    assert response["metadata"]["engine"] == "kairos_des"

    result = response["result"]
    counts = result["counts"]
    assert len(counts) == len(times)
    assert all(type(count) is int and 0 <= count <= options["population"] for count in counts)
    assert counts[0] == 0 and counts[-1] > 0
    assert all(left <= right for left, right in pairwise(counts))

    # Independent unit conversion: each agent represents m / population units.
    expected_levels = [parameters["m"] * count / options["population"] for count in counts]
    assert all(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        for actual, expected in zip(result["values"], expected_levels, strict=True)
    )
    trajectory = pl.DataFrame({"time": times, "adopters": counts, "level": result["values"]})
    assert trajectory["time"].to_list() == times

    # Merely asking for fewer observations must not restart the event process.
    sparse = execute("simulate_model", "bass", **{**request, "inputs": {"time": times[2:]}})["result"]
    sparse_grid_equal = sparse["counts"] == counts[2:] and sparse["values"] == result["values"][2:]
    assert sparse_grid_equal

    # With no innovators and no initial adopters, imitation has no source.
    no_innovators = execute("simulate_model", "bass", **{**request, "parameters": {**parameters, "p": 0.0}})["result"]
    assert no_innovators["counts"] == [0] * len(times)

    try:
        execute("simulate_model", "bass", **{**request, "simulation_options": {**options, "engine": "unknown"}})
    except NativeError as error:
        unknown_engine_rejected = error.code == "invalid_request"
    else:
        unknown_engine_rejected = False
    assert unknown_engine_rejected, "An unsupported engine must fail instead of selecting a deterministic curve."

    print(
        json.dumps(
            {
                "counts": trajectory["adopters"].to_list(),
                "levels": trajectory["level"].to_list(),
                "replay_equal": replay_equal,
                "sparse_grid_equal": sparse_grid_equal,
                "unknown_engine_rejected": unknown_engine_rejected,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
