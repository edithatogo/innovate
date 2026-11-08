"""Measure paired percentile coverage against the directed-chain oracle."""

import json
from math import exp, expm1
from statistics import mean

from innovate import execute


def measure(pair_count: int) -> dict[str, int | float]:
    """Use nested seed sets to compare ensemble sizes on the same design."""
    trials = 500
    survival = exp(-0.0005)
    success = -expm1(-0.0005)
    expected = 2 - survival**4000 - survival**3000 - 3000 * success * survival**2999
    covered = 0
    widths = []
    for trial in range(trials):
        scenarios = []
        pairs = []
        for pair in range(pair_count):
            baseline = f"{pair}-baseline"
            alternative = f"{pair}-alternative"
            pairs.append({"baseline": baseline, "alternative": alternative})
            for scenario_id, imitation in ((baseline, 0.0), (alternative, 0.5)):
                scenarios.append(
                    {
                        "id": scenario_id,
                        "model_key": "bass",
                        "simulation": {
                            "inputs": {"time": [0.0, 5.0]},
                            "parameters": {"p": 0.0, "q": imitation, "m": 3.0},
                            "simulation_options": {
                                "engine": "kairos_network",
                                "population": 3,
                                "seed": 60000 + 64 * trial + pair,
                                "edges": [[0, 1], [1, 2]],
                                "directed": True,
                                "initial_agents": [0],
                                "capacity": 1,
                                "service_ticks": 1000,
                                "ticks_per_unit": 1000,
                            },
                        },
                    }
                )
        response = execute(
            "execute_scenario_grid",
            scenarios=scenarios,
            comparisons=[
                {
                    "pairs": pairs,
                    "interval_mass": 0.9,
                    "require_distinct_pair_seeds": True,
                    "bootstrap": {"draws": 1000, "seed": 100000 + trial},
                }
            ],
        )
        comparison = response["result"]["comparisons"][0]
        if comparison["error"] is not None:
            raise RuntimeError(comparison["error"])
        lower, upper = comparison["result"]["bootstrap"]["mean_interval"]
        covered += lower <= expected <= upper
        widths.append(upper - lower)
    return {
        "pairs": pair_count,
        "trials": trials,
        "covered": covered,
        "analytic_effect": expected,
        "mean_interval_width": mean(widths),
    }


if __name__ == "__main__":
    print(json.dumps([measure(32), measure(64)], indent=2, allow_nan=False))
