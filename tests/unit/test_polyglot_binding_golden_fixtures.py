"""Tests for polyglot binding golden fixtures."""

from __future__ import annotations

import json
from pathlib import Path

GOLDEN_PATH = Path("docs/source/_static/binding_golden_fixtures.json")
RUST_GOLDEN_PATH = Path("bindings/rust/inst/binding_golden_fixtures.json")
REQUIRED_OPERATIONS = {
    "discover_models",
    "fit_model",
    "predict_model",
    "simulate_model",
    "summarize_model",
    "diagnose_model",
}


def _fixtures() -> dict[str, object]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_binding_golden_fixtures_cover_promoted_operations() -> None:
    """Golden fixtures should cover every promoted canonical operation."""
    payload = _fixtures()

    assert payload["schema_version"] == "binding_golden_fixtures.v1"
    assert payload["kernel_schema_version"] == "1.0"
    fixtures = {case["operation"]: case for case in payload["fixtures"]}
    assert set(fixtures) == REQUIRED_OPERATIONS
    for case in fixtures.values():
        assert case["request"]["schema_version"] == "1.0"
        assert case["request"]["operation"] == case["operation"]
        assert case["expected"]["ok"] is True
        assert case["tolerance"]["absolute"] <= 1e-6
        assert case["tolerance"]["relative"] <= 1e-6


def test_binding_golden_fixtures_have_round_trip_payloads() -> None:
    """Fixture payloads should include operation, response, and error round-trip cases."""
    payload = _fixtures()

    assert set(payload["payload_round_trips"]) >= {
        "KernelRequest",
        "KernelResponse",
        "KernelError",
        "KernelArrayPayload",
        "KernelTablePayload",
    }
    error_fixture = payload["error_fixture"]
    assert error_fixture["code"] == "unsupported_operation"
    assert error_fixture["retryable"] is False


def test_rust_binding_uses_same_golden_fixture_bytes() -> None:
    """Rust binding tests should consume the same golden fixture payload."""
    assert RUST_GOLDEN_PATH.read_bytes() == GOLDEN_PATH.read_bytes()
