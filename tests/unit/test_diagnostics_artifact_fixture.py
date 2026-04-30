"""Cross-language fixture checks for diagnostics artifact payloads."""

from __future__ import annotations

import json
from pathlib import Path


def test_diagnostics_artifact_fixture_matches_binding_contract() -> None:
    """Representative diagnostics artifacts should stay stable for thin bindings."""
    fixture_path = Path(__file__).parents[1] / "fixtures" / "diagnostics_artifact_payload.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    diagnostics = payload["result"]["diagnostics"]
    artifacts = diagnostics["artifacts"]

    assert payload["schema_version"] == "1.0"
    assert diagnostics["support_level"] == "supported"
    assert artifacts["schema_version"] == "1.0"
    assert artifacts["xla"]["eligible"] is False
    assert artifacts["artifacts"]["residuals"]["columns"] == [
        "index",
        "residual",
        "standardized_residual",
    ]
    assert artifacts["artifacts"]["model_comparison"]["columns"] == ["metric", "value"]
