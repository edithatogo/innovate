"""Prepare raw JSON requests for the Julia and R parity checks."""

import json
import sys
from pathlib import Path

data = json.loads((Path(__file__).parent / "fixtures.json").read_text(encoding="utf-8"))
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
expected = []
for index, case in enumerate(data["native"]):
    artifact = case["artifact"]
    request = {
        "schema_version": "2.0",
        "operation": "import_scenario_artifact",
        "payload": {"artifact": artifact},
    }
    (output / f"{index:02}-import.json").write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
    (output / f"{index:02}-replay.json").write_text(
        json.dumps(artifact["replay_request"], allow_nan=False), encoding="utf-8"
    )
    expected.append({"id": case["id"], "expected": case["expected"]})
for index, artifact in enumerate(data["legacy"]):
    request = {
        "schema_version": "2.0",
        "operation": "import_scenario_artifact",
        "payload": {"artifact": artifact},
    }
    (output / f"legacy-{index:02}.json").write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
(output / "expected.json").write_text(json.dumps(expected, allow_nan=False), encoding="utf-8")
print(f"Prepared {len(expected)} native cases and {len(data['legacy'])} legacy artifacts.")
