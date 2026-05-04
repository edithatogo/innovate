"""Contract checks for HEOR process-mining ecosystem fixtures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

FIXTURE_ROOT = Path("specs/ecosystem/process/fixtures/event_log_v1")
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_process_mining_manifest_defines_portable_documented_contract() -> None:
    """The process fixture should be versioned and independent of PM4Py."""
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["schema_version"] == "innovate.ecosystem.process.event_log.v1"
    assert manifest["fixture_id"] == "heor_process_event_log_smoke_v1"
    assert manifest["artifact_type"] == "process_mining.event_log_bundle"
    assert manifest["promotion_stage"] == "documented"
    assert manifest["runtime_adapter_status"] == "future_work"
    assert manifest["engine_dependency"] == "none"
    assert manifest["reference_candidate"] == "PM4Py"
    assert manifest["pm4py_required"] is False
    assert manifest["dependency_policy"]["innovate_base_import_required"] is False
    assert manifest["dependency_policy"]["pm4py_required"] is False
    assert manifest["dependency_policy"]["optional_extra_required"] is False
    assert "require pickle" in manifest["consumer"]["must_not"]
    assert "require private Python classes" in manifest["consumer"]["must_not"]


def test_process_mining_payloads_are_small_and_internally_consistent() -> None:
    """Fixture payloads should be deterministic, ordered, and cross-linked."""
    manifest = _load_json(MANIFEST_PATH)
    event_payload = manifest["payloads"]["event_log"]
    event_rows = _load_csv(FIXTURE_ROOT / event_payload["path"])
    pathway_rows = _load_csv(FIXTURE_ROOT / manifest["payloads"]["pathway_discovery"]["path"])
    bottleneck_rows = _load_csv(FIXTURE_ROOT / manifest["payloads"]["bottleneck_summary"]["path"])
    conformance = _load_json(FIXTURE_ROOT / manifest["payloads"]["conformance_summary"]["path"])

    assert len(event_rows) == event_payload["row_count"] == 8
    assert len(pathway_rows) == manifest["payloads"]["pathway_discovery"]["row_count"] == 1
    assert len(bottleneck_rows) == manifest["payloads"]["bottleneck_summary"]["row_count"] == 3
    assert list(event_rows[0]) == event_payload["required_columns"]
    assert conformance["schema_version"] == "innovate.ecosystem.process.conformance.v1"
    assert conformance["fixture_id"] == manifest["fixture_id"]
    assert conformance["fitness"] == 1.0
    assert conformance["precision"] == 1.0

    assert event_rows == sorted(
        event_rows,
        key=lambda row: (row["case_id"], int(row["event_index"]), row["event_time"]),
    )
    assert {row["case_id"] for row in event_rows} == {"case_001", "case_002"}
    assert {row["activity"] for row in event_rows} == {
        "referral_received",
        "triage_completed",
        "assessment_completed",
        "intervention_started",
    }

    pathway = pathway_rows[0]
    assert pathway["variant_id"] == conformance["reference_variant_id"]
    assert (
        pathway["activity_sequence"] == "referral_received>triage_completed>assessment_completed>intervention_started"
    )

    for row in bottleneck_rows:
        assert float(row["mean_wait_days"]) >= 0
        assert float(row["p95_wait_days"]) >= float(row["mean_wait_days"])


def test_process_mining_manifest_records_interface_decisions() -> None:
    """CLI and MCP decisions should be explicit before adapter work starts."""
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["cli_surface"]["status"] == "planned_before_runtime_adapter"
    assert manifest["cli_surface"]["commands"] == [
        "validate-event-log",
        "summarize-pathway",
        "export-conformance-summary",
    ]
    assert manifest["mcp_surface"]["status"] == "deferred"
    assert "agent-queryable" in manifest["mcp_surface"]["rationale"]


def test_process_mining_docs_link_fixture_and_scope_limits() -> None:
    """Ecosystem docs should expose process fixture scope and dependency policy."""
    process_readme = Path("specs/ecosystem/process/README.md").read_text(encoding="utf-8")
    fixture_readme = Path("specs/ecosystem/process/fixtures/README.md").read_text(encoding="utf-8")
    ecosystem_readme = Path("specs/ecosystem/README.md").read_text(encoding="utf-8")
    strategy = Path("docs/ecosystem/module_incubation_strategy.md").read_text(encoding="utf-8")

    for text in (process_readme, fixture_readme, ecosystem_readme, strategy):
        assert "process/fixtures/event_log_v1/manifest.json" in text or "event_log_v1/manifest.json" in text
        assert "PM4Py" in text

    normalized_strategy = " ".join(strategy.split()).lower()
    assert "cli support is planned before adapter implementation" in normalized_strategy
    assert "mcp remains deferred" in normalized_strategy
    assert "rather than a base dependency" in normalized_strategy
