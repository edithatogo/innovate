"""Check native synthetic scenarios, artifact round trips and deterministic replay.

Reuse the scenario payloads from the maintained tutorial. JSON and Arrow output
are written to a temporary directory and removed when the example exits. The
example uses no external datasets, services or host-language numerical fallback.
"""

import json
import math
import tempfile
from pathlib import Path

import polars as pl

import innovate


def main() -> None:
    """Execute four synthetic scenarios and check their scientific invariants."""
    common = {
        "description": "Synthetic scenario example",
        "time_horizon": 5,
        "time_unit": "year",
        "reference_year": 2026,
    }
    policy = {
        "id": "pause-innovation",
        "model_key": "bass",
        "specification": {
            **common,
            "scenario_type": "intervention",
            "name": "Pause innovation",
            "market_size": 100,
            "initial_adoption": 0,
            "intervention_type": "innovation_delta",
            "intervention_start_time": 0,
            "intervention_magnitude": -0.2,
        },
        "simulation": {
            "inputs": {"time": [0, 5]},
            "parameters": {"p": 0.2, "q": 0},
            "simulation_options": {
                "engine": "kairos_abm",
                "seed": 42,
                "population": 100,
            },
        },
    }
    baseline = {
        **policy,
        "id": "baseline",
        "specification": {
            **common,
            "scenario_type": "baseline",
            "name": "Baseline",
            "market_size": 100,
            "initial_adoption": 0,
        },
    }
    competition = {
        "id": "two-products",
        "model_key": "multi_product",
        "specification": {
            **common,
            "scenario_type": "competition",
            "name": "Two products",
            "market_size": 200,
            "initial_adoption": 0.5,
            "num_competitors": 2,
            "competitor_names": ["A", "B"],
            "market_share_initial": [0.25, 0.75],
        },
        "simulation": {
            "inputs": {"time": [0, 5]},
            "parameters": {
                "p1": 0,
                "p2": 0,
                "q1": 0,
                "q2": 0,
                "m1": 100,
                "m2": 100,
                "alpha_1_2": 0,
                "alpha_2_1": 0,
            },
        },
    }
    substitution = {
        "id": "replacement",
        "model_key": "fixed_pool_replacement",
        "specification": {
            **common,
            "scenario_type": "substitution",
            "name": "Fixed-pool replacement",
            "market_size": 200,
            "initial_adoption": 0.5,
            "incumbent_name": "A",
            "entrant_name": "B",
            "substitution_rate": 0.5,
        },
        "simulation": {"inputs": {"time": [0, 2]}},
    }

    response = innovate.execute(
        "execute_scenario_grid",
        scenarios=[baseline, policy, competition, substitution],
        comparisons=[{"baseline": "baseline", "alternative": "pause-innovation"}],
        apply_specifications=True,
        include_execution_metadata=True,
    )
    executions = response["result"]["executions"]
    for row in executions:
        if row["error"] is not None:
            raise RuntimeError(row["error"])

    assert executions[1]["result"]["counts"] == [0, 0]
    comparison = response["result"]["comparisons"][0]
    if comparison["error"] is not None:
        raise RuntimeError(comparison["error"])
    baseline_final = executions[0]["result"]["values"][-1]
    assert baseline_final > 0
    assert comparison["result"]["difference"][-1] == -baseline_final
    assert executions[2]["result"]["rows"] == [[25, 75], [25, 75]]
    incumbent, entrant = executions[3]["result"]["rows"][-1]
    assert math.isclose(incumbent, 100 * math.exp(-1))
    assert math.isclose(incumbent + entrant, 100)

    with tempfile.TemporaryDirectory(prefix="innovate-scenario-example-") as directory:
        json_path = Path(directory) / "scenario-grid.json"
        arrow_path = Path(directory) / "scenario-executions.arrow"
        json_path.write_text(json.dumps(response, indent=2, allow_nan=False), encoding="utf-8")
        assert json.loads(json_path.read_text(encoding="utf-8")) == response
        innovate.scenario_executions(response).write_ipc(arrow_path)
        table = pl.read_ipc(arrow_path)
        assert table["scenario_id"].to_list() == ["baseline", "pause-innovation", "two-products", "replacement"]
        saved_rows = [json.loads(value) for value in table["execution_json"]]
        assert saved_rows == executions

        replayed = 0
        for row in saved_rows:
            imported = innovate.execute("import_scenario_artifact", artifact=row["artifact"])["result"]
            if not imported["replay_available"]:
                raise RuntimeError(imported["replay_reason"])
            request = imported["artifact"]["replay_request"]
            replay = innovate.execute(request["operation"], **request["payload"])
            assert replay["result"]["executions"][0]["result"] == row["result"]
            replayed += 1

    assert len(executions) == replayed == 4
    print(
        json.dumps(
            {
                "policy_counts": executions[1]["result"]["counts"],
                "replacement_total": round(incumbent + entrant, 6),
                "replayed": replayed,
                "scenario_ids": table["scenario_id"].to_list(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
