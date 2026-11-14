"""Check an exact adoption pause, equivalent forcing and tracked ABM agents.

The synthetic covariate cancels both Bass rates until time three. A later event
raises innovation at time six. These piecewise-constant simulation controls are
distinct from fitted covariate effects or empirical intervention estimates.
"""

import json
from itertools import pairwise

from innovate import NativeError, execute


def main() -> None:
    """Exercise covariates, additive events, agent conservation and input guards."""
    times = [0.0, 1.0, 3.0, 5.0, 6.0, 10.0]
    parameters = {"p": 0.2, "q": 0.3, "m": 100.0}
    options = {
        "engine": "kairos_abm",
        "seed": 42,
        "population": 100,
        "ticks_per_unit": 1_000_000,
        "covariate": {"time": [0.0, 3.0], "values": [-1.0, 0.0], "beta_p": 0.2, "beta_q": 0.3},
        "events": [{"time": 6.0, "delta_p": 0.1, "delta_q": 0.0}],
    }
    request = {"inputs": {"time": times}, "parameters": parameters, "simulation_options": options}
    response = execute("simulate_model", "bass", **request)
    result = response["result"]
    counts = result["counts"]
    assert len(counts) == len(times)

    # Before time three, p + beta_p*x and q + beta_q*x are both exactly zero.
    # At the boundary, adoption still needs at least one positive scheduler tick.
    assert counts[:3] == [0, 0, 0]
    assert counts[3] > 0
    assert all(0 <= count <= options["population"] for count in counts)
    assert all(left <= right for left, right in pairwise(counts))
    agents = result["agents"]
    assert len(agents) == options["population"]
    assert len({(agent["index"], agent["generation"]) for agent in agents}) == options["population"]
    assert sum(agent["adopted"] for agent in agents) == counts[-1]
    assert response == execute("simulate_model", "bass", **request)

    # The same rate schedule can be expressed entirely through additive events.
    # Identical boundaries and rates should consume the same seeded streams.
    event_options = {key: value for key, value in options.items() if key != "covariate"}
    event_options["events"] = [
        {"time": 0.0, "delta_p": -0.2, "delta_q": -0.3},
        {"time": 3.0, "delta_p": 0.2, "delta_q": 0.3},
        {"time": 6.0, "delta_p": 0.1, "delta_q": 0.0},
    ]
    equivalent = execute("simulate_model", "bass", **{**request, "simulation_options": event_options})["result"]
    equivalent_forcing = all(equivalent[key] == result[key] for key in ("counts", "values", "agents"))
    assert equivalent_forcing

    # A negative rate and two distinct changes rounded to the same tick must fail.
    invalid_options = [
        {**options, "covariate": {**options["covariate"], "beta_p": 2.0}},
        {
            **options,
            "ticks_per_unit": 1,
            "events": [
                {"time": 0.1, "delta_p": 0.0, "delta_q": 0.0},
                {"time": 0.2, "delta_p": 0.0, "delta_q": 0.0},
            ],
        },
    ]
    invalid_forcing_rejected = 0
    for invalid in invalid_options:
        try:
            execute("simulate_model", "bass", **{**request, "simulation_options": invalid})
        except NativeError as error:
            assert error.code == "invalid_request"
            invalid_forcing_rejected += 1
        else:
            raise AssertionError("Invalid rates or colliding changes were silently accepted.")
    assert invalid_forcing_rejected == len(invalid_options)

    print(
        json.dumps(
            {
                "counts": counts,
                "equivalent_forcing": equivalent_forcing,
                "invalid_forcing_rejected": invalid_forcing_rejected,
                "tracked_agents": len(agents),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
