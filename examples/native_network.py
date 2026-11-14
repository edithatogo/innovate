"""Check contagion, service completion and FIFO resource accounting.

One synthetic time unit equals one scheduler tick so the five-tick service time
is visible. The separate queue fixture uses a very high innovation rate to expose
the scheduler's minimum one-tick delay; it is not a calibrated adoption model.
"""

import json

from innovate import NativeError, execute


def main() -> None:
    """Verify graph-dependent spread, independent service timing and graph errors."""
    times = [0.0, 5.0, 20.0, 100.0]
    parameters = {"p": 0.0, "q": 3.0, "m": 3.0}
    options = {
        "engine": "kairos_network",
        "seed": 42,
        "population": 3,
        "edges": [[0, 1], [1, 2]],
        "initial_agents": [0],
        "capacity": 1,
        "service_ticks": 5,
        "ticks_per_unit": 1,
    }
    request = {"inputs": {"time": times}, "parameters": parameters, "simulation_options": options}
    response = execute("simulate_model", "bass", **request)
    chain = response["result"]
    assert response == execute("simulate_model", "bass", **request)
    assert chain["counts"][0] == 1 and chain["counts"][-1] == options["population"]
    assert all(agent["state"] == "adopted" for agent in chain["agents"])

    # With p=0, node two can request only after its sole neighbor one adopts.
    completions = {event["agent"]: event["tick"] for event in chain["completions"]}
    requests = {event["agent"]: event["tick"] for event in chain["requests"]}
    assert set(completions) == set(requests) == {1, 2}
    assert requests[2] > completions[1]
    assert chain["counts"][-1] == len(options["initial_agents"]) + len(completions)
    disconnected = execute("simulate_model", "bass", **{**request, "simulation_options": {**options, "edges": []}})[
        "result"
    ]
    assert disconnected["counts"] == [1] * len(times)
    assert disconnected["requests"] == disconnected["completions"] == []

    # A second fixture creates demand without peers and queues three agents.
    queue_options = {**options, "edges": [], "initial_agents": []}
    queued = execute(
        "simulate_model",
        "bass",
        inputs={"time": [0.0, 5.0, 20.0]},
        parameters={"p": 1e12, "q": 0.0, "m": 3.0},
        simulation_options=queue_options,
    )["result"]
    assert queued["counts"] == [0, 0, 3], "A pending request is not yet adoption."
    assert len(queued["requests"]) == len(queued["completions"]) == 3
    free_at = 0
    expected_occupied_ticks = 0
    expected_wait_ticks = 0
    for demand, completion in zip(queued["requests"], queued["completions"], strict=True):
        # Independent single-server FIFO recurrence; no native queue helper used.
        start = max(demand["tick"], free_at)
        free_at = start + queue_options["service_ticks"]
        assert completion["agent"] == demand["agent"]
        assert completion["tick"] == free_at
        expected_occupied_ticks += queue_options["service_ticks"]
        expected_wait_ticks += start - demand["tick"]
    metrics = queued["resource_metrics"]
    assert metrics["peak_in_service"] == queue_options["capacity"]
    assert metrics["occupied_slot_ticks"] == expected_occupied_ticks
    assert metrics["admitted_wait_ticks"] == expected_wait_ticks
    assert metrics["configured_slot_ticks"] == metrics["horizon_ticks"] * queue_options["capacity"]
    assert metrics["idle_slot_ticks"] + metrics["occupied_slot_ticks"] == metrics["configured_slot_ticks"]
    assert metrics["over_capacity_slot_ticks"] == 0

    try:
        execute("simulate_model", "bass", **{**request, "simulation_options": {**options, "edges": [[0, 0]]}})
    except NativeError as error:
        invalid_edges_rejected = error.code == "invalid_request"
    else:
        invalid_edges_rejected = False
    assert invalid_edges_rejected, "An undirected self-loop must fail the native graph contract."

    print(
        json.dumps(
            {
                "chain_counts": chain["counts"],
                "disconnected_counts": disconnected["counts"],
                "fifo_completions": [event["tick"] for event in queued["completions"]],
                "invalid_edges_rejected": invalid_edges_rejected,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
