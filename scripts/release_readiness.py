"""Build fail-closed release-readiness reports from committed evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = Path("docs/source/_static/release_readiness_contract.json")
DEFAULT_EVIDENCE_ROOT = Path("docs/source/_static/release_readiness/evidence")
VALID_STATUS_VALUES = {
    "blocked",
    "deferred",
    "fail",
    "manual",
    "missing",
    "pass",
    "release_candidate",
    "release_ready",
    "stale",
    "unknown",
}
REQUIRED_EVIDENCE_IDS = {
    "python_tests",
    "coverage",
    "mutation_sampling",
    "type_checks",
    "lint_format",
    "docs_build",
    "rust_tests",
    "binding_smoke",
    "package_dry_run",
    "security_audit",
    "sbom",
    "license_inventory",
    "provenance",
    "checksums",
    "reproducibility",
    "compatibility",
}
PASSING_STATUSES = {"pass"}
FAILING_STATUSES = {"fail", "blocked", "unknown", "manual", "deferred"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load and validate the committed release-readiness contract."""
    contract = json.loads(_resolve(path).read_text(encoding="utf-8"))
    evidence_ids = {entry["id"] for entry in contract["required_evidence"]}
    status_values = set(contract["status_values"])

    if evidence_ids != REQUIRED_EVIDENCE_IDS:
        missing = sorted(REQUIRED_EVIDENCE_IDS - evidence_ids)
        extra = sorted(evidence_ids - REQUIRED_EVIDENCE_IDS)
        raise ValueError(f"Release-readiness evidence ids drifted: missing={missing}, extra={extra}")
    if status_values != VALID_STATUS_VALUES:
        raise ValueError("Release-readiness status values drifted from evaluator constants")

    return contract


def _load_evidence(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Evidence artifact must be a JSON object: {path}")
    return data


def _artifact_age_days(path: Path, now: float) -> float:
    return (now - path.stat().st_mtime) / 86_400


def evaluate_evidence(
    *,
    contract: dict[str, Any],
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    now: float | None = None,
) -> dict[str, Any]:
    """Evaluate required evidence and fail closed when anything is absent or stale."""
    root = _resolve(evidence_root)
    current_time = time.time() if now is None else now
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    stale: list[dict[str, Any]] = []
    failing: list[dict[str, Any]] = []
    counts = {"pass": 0, "missing": 0, "stale": 0, "fail": 0}

    for requirement in contract["required_evidence"]:
        artifact_path = root / requirement["artifact"]
        check = {
            "id": requirement["id"],
            "owner": requirement["owner"],
            "lane": requirement["lane"],
            "artifact": str(artifact_path.relative_to(ROOT) if artifact_path.is_relative_to(ROOT) else artifact_path),
            "producer": requirement["producer"],
            "status": "missing",
        }

        if not artifact_path.is_file():
            missing.append(requirement["id"])
            counts["missing"] += 1
            checks.append(check)
            continue

        data = _load_evidence(artifact_path)
        status = str(data.get("status", "unknown"))
        age_days = _artifact_age_days(artifact_path, current_time)
        check.update(
            {
                "status": status,
                "freshness_days": requirement["freshness_days"],
                "age_days": round(age_days, 3),
                "summary": data.get("summary", ""),
            }
        )

        if age_days > requirement["freshness_days"] or status == "stale":
            check["status"] = "stale"
            stale.append(check)
            counts["stale"] += 1
        elif status in PASSING_STATUSES:
            counts["pass"] += 1
        elif status in FAILING_STATUSES or status not in VALID_STATUS_VALUES:
            failing.append(check)
            counts["fail"] += 1
        else:
            failing.append(check)
            counts["fail"] += 1

        checks.append(check)

    blocked = bool(missing or stale or failing)
    return {
        "schema_version": 1,
        "contract_path": str(CONTRACT_PATH),
        "evidence_root": str(evidence_root),
        "overall_status": "blocked" if blocked else "release_ready",
        "release_state": "release_candidate" if blocked else "release_ready",
        "status_counts": counts,
        "missing_evidence": missing,
        "stale_evidence": stale,
        "failing_evidence": failing,
        "checks": checks,
    }


def build_readiness_report(
    *,
    contract_path: Path = CONTRACT_PATH,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
) -> dict[str, Any]:
    """Build the default release-readiness report from repository artifacts."""
    contract = load_contract(contract_path)
    return evaluate_evidence(contract=contract, evidence_root=evidence_root)


def render_text(report: dict[str, Any]) -> str:
    """Render a concise human-readable release-readiness summary."""
    lines = [
        "Release readiness",
        f"- overall_status: {report['overall_status']}",
        f"- release_state: {report['release_state']}",
    ]
    for status, count in report["status_counts"].items():
        lines.append(f"- {status}: {count}")
    if report["missing_evidence"]:
        lines.append(f"- missing_evidence: {', '.join(report['missing_evidence'])}")
    if report["stale_evidence"]:
        lines.append("- stale_evidence: " + ", ".join(item["id"] for item in report["stale_evidence"]))
    if report["failing_evidence"]:
        lines.append("- failing_evidence: " + ", ".join(item["id"] for item in report["failing_evidence"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for local and CI release-readiness checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    args = parser.parse_args(argv)

    report = build_readiness_report(evidence_root=args.evidence_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))

    return 0 if report["overall_status"] == "release_ready" else 1


if __name__ == "__main__":
    sys.exit(main())
