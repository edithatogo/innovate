"""Verify retained historical scenarios against an installed native Innovate build."""

import copy
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import innovate
import innovate._native

fixtures = json.loads((Path(__file__).parent / "fixtures.json").read_text(encoding="utf-8"))
assert len(fixtures["legacy"]) == 5
for artifact in fixtures["legacy"]:
    imported = innovate.execute("import_scenario_artifact", artifact=artifact)["result"]
    assert imported["format"] == "legacy_scenario_envelope"
    assert imported["artifact"] == artifact
    assert imported["artifact"]["seed"] == 2**64 - 1
    assert not imported["replay_available"]
    assert imported["replay_reason"] == "legacy artifact has no simulation payload"
    invalid = copy.deepcopy(artifact)
    invalid["timestamp"] = "2026-01-01"
    try:
        innovate.execute("import_scenario_artifact", artifact=invalid)
    except innovate.NativeError as error:
        assert error.code == "invalid_request"
    else:
        raise AssertionError("Legacy artifact with an invalid timestamp was accepted")

assert len(fixtures["native"]) == 24
for case in fixtures["native"]:
    imported = innovate.execute("import_scenario_artifact", artifact=case["artifact"])["result"]
    assert imported["replay_available"], (case["id"], imported["replay_reason"])
    request = imported["artifact"]["replay_request"]
    response = innovate.execute(request["operation"], **request["payload"])
    row = response["result"]["executions"][0]
    assert row["error"] is None, (case["id"], row["error"])
    for key, expected in case["expected"].items():
        assert row["result"][key] == expected, (case["id"], key)

times = [0.0, 1.0, 2.0, 5.0]
scenarios = []
for initial in [0.0, 0.1, 0.5, 1.0]:
    for hazard in [0.0, 0.1, 0.5, 1.0]:
        scenarios.append(
            {
                "id": f"initial-{initial}-hazard-{hazard}",
                "model_key": "fixed_pool_replacement",
                "specification": {
                    "scenario_type": "substitution",
                    "name": "replacement",
                    "description": "Analytic fixed-pool fixture",
                    "time_horizon": 5,
                    "time_unit": "years",
                    "reference_year": 2026,
                    "market_size": 200,
                    "initial_adoption": initial,
                    "incumbent_name": "A",
                    "entrant_name": "B",
                    "substitution_rate": hazard,
                },
                "simulation": {"inputs": {"time": times}},
            }
        )
response = innovate.execute(
    "execute_scenario_grid",
    scenarios=scenarios,
    apply_specifications=True,
    include_execution_metadata=True,
)
rows = response["result"]["executions"]
assert len(rows) == len(scenarios) == 16
for scenario, row in zip(scenarios, rows, strict=True):
    assert row["error"] is None, (scenario["id"], row["error"])
    spec = scenario["specification"]
    pool = spec["market_size"] * spec["initial_adoption"]
    for time, values in zip(times, row["result"]["rows"], strict=True):
        incumbent, entrant = values
        reference = pool * math.exp(-spec["substitution_rate"] * time)
        assert math.isclose(incumbent, reference, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(incumbent + entrant, pool, rel_tol=1e-12, abs_tol=1e-12)
    imported = innovate.execute("import_scenario_artifact", artifact=row["artifact"])["result"]
    assert imported["replay_available"], imported["replay_reason"]
    request = imported["artifact"]["replay_request"]
    replay = innovate.execute(request["operation"], **request["payload"])
    assert replay["result"]["executions"][0]["result"] == row["result"]

library = Path(innovate._native.__file__).resolve()
print("Python", sys.version)
print("Innovate", importlib.metadata.version("innovate"), "from", innovate.__file__)
print("Native extension", library)
print("Native extension SHA-256", hashlib.sha256(library.read_bytes()).hexdigest())
print("Five legacy imports, 24 historical replays and 16 analytic substitution cases passed.")
