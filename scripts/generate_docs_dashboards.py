"""Generate evidence-backed Starlight dashboard artifacts and pages."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "docs/source/_static"
ASTRO_DOCS = ROOT / "docs/astro-site/src/content/docs"
OUTPUT = STATIC_ROOT / "astro_starlight/release_maturity_dashboard.json"
OBSERVABILITY_OUTPUT = STATIC_ROOT / "astro_starlight/observability_maintenance.json"

SOURCE_ARTIFACTS = {
    "release_readiness": STATIC_ROOT / "release_readiness_contract.json",
    "rust_ownership": STATIC_ROOT / "rust_full_ownership_gate.json",
    "registry_state": STATIC_ROOT / "registry_submission_inventory.json",
    "binding_conformance": STATIC_ROOT / "binding_conformance_inventory.json",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def build_dashboard() -> dict[str, Any]:
    """Build a release maturity dashboard from source evidence artifacts."""
    evidence_date = date.today().isoformat()
    release = _read_json(SOURCE_ARTIFACTS["release_readiness"])
    rust = _read_json(SOURCE_ARTIFACTS["rust_ownership"])
    registry = _read_json(SOURCE_ARTIFACTS["registry_state"])
    bindings = _read_json(SOURCE_ARTIFACTS["binding_conformance"])

    target_status_counts = Counter(target["submission_status"] for target in registry["targets"])
    binding_status_counts = Counter(binding["status"] for binding in bindings["bindings"])

    return {
        "schema_version": 1,
        "generated_by_track": "production_docs_observability_20260614",
        "generated_at": f"{evidence_date}T00:00:00Z",
        "evidence_date": evidence_date,
        "staleness": {
            "max_age_days": 30,
            "status": "fresh",
            "source": "generated_at",
        },
        "source_artifacts": {key: _rel(path) for key, path in SOURCE_ARTIFACTS.items()},
        "cards": [
            {
                "id": "release_readiness",
                "title": "Release readiness",
                "status": "release_candidate_evidence_defined",
                "source": _rel(SOURCE_ARTIFACTS["release_readiness"]),
                "metrics": {
                    "required_evidence_count": len(release["required_evidence"]),
                    "status_values": release["status_values"],
                },
                "summary": "Release readiness has a defined evidence contract; release-ready status still depends on current CI and release-lane evidence.",
            },
            {
                "id": "rust_ownership",
                "title": "Rust ownership",
                "status": "full_rust_ownership_not_claimed",
                "source": _rel(SOURCE_ARTIFACTS["rust_ownership"]),
                "metrics": {
                    "blocking_model_family_count": len(rust["blocking_model_families"]),
                    "blocking_payload_shape_count": len(rust["blocking_payload_shapes"]),
                    "decision": rust["decision"],
                },
                "summary": rust["claim_language"],
            },
            {
                "id": "registry_state",
                "title": "Registry state",
                "status": "mixed_external_acceptance",
                "source": _rel(SOURCE_ARTIFACTS["registry_state"]),
                "metrics": {
                    "package_targets": len(registry["package_targets"]),
                    "hpc_targets": len(registry["hpc_targets"]),
                    "submission_status_counts": dict(sorted(target_status_counts.items())),
                },
                "summary": "Registry evidence includes published, submitted, deferred, and maintainer-review states; not all external registries are accepted.",
            },
            {
                "id": "binding_conformance",
                "title": "Binding conformance",
                "status": "supported_bindings_documented",
                "source": _rel(SOURCE_ARTIFACTS["binding_conformance"]),
                "metrics": {
                    "binding_count": len(bindings["bindings"]),
                    "binding_status_counts": dict(sorted(binding_status_counts.items())),
                    "kernel_schema_version": bindings["kernel_schema_version"],
                },
                "summary": "Polyglot binding conformance is documented across supported bindings with package checks and evidence paths.",
            },
        ],
        "claim_guardrails": {
            "external_acceptance": "Do not claim all registries accepted.",
            "rust_ownership": "Do not claim full Rust ownership.",
        },
    }


def build_observability(dashboard: dict[str, Any]) -> dict[str, Any]:
    """Build support and maintenance policy route evidence."""
    evidence_date = dashboard["evidence_date"]
    pages = [
        {
            "id": "package_health",
            "title": "Package Health",
            "current_route": "/maintainers/package-health/",
            "latest_route": "/latest/maintainers/package-health/",
            "evidence_links": [
                _rel(OUTPUT),
                _rel(SOURCE_ARTIFACTS["registry_state"]),
                _rel(SOURCE_ARTIFACTS["binding_conformance"]),
            ],
        },
        {
            "id": "compatibility",
            "title": "Compatibility Policy",
            "current_route": "/maintainers/compatibility/",
            "latest_route": "/latest/maintainers/compatibility/",
            "evidence_links": [
                _rel(OUTPUT),
                _rel(SOURCE_ARTIFACTS["binding_conformance"]),
            ],
        },
        {
            "id": "deprecation",
            "title": "Deprecation Policy",
            "current_route": "/maintainers/deprecation/",
            "latest_route": "/latest/maintainers/deprecation/",
            "evidence_links": [
                _rel(OUTPUT),
                _rel(SOURCE_ARTIFACTS["release_readiness"]),
            ],
        },
        {
            "id": "support",
            "title": "Support Policy",
            "current_route": "/maintainers/support/",
            "latest_route": "/latest/maintainers/support/",
            "evidence_links": [
                _rel(OUTPUT),
                "docs/source/_static/astro_starlight/production_docs_verification.json",
            ],
        },
        {
            "id": "maintenance",
            "title": "Maintenance Policy",
            "current_route": "/maintainers/maintenance/",
            "latest_route": "/latest/maintainers/maintenance/",
            "evidence_links": [
                _rel(OUTPUT),
                "docs/source/_static/astro_starlight/production_docs_verification.json",
            ],
        },
    ]
    return {
        "schema_version": 1,
        "generated_by_track": "production_docs_observability_20260614",
        "generated_at": f"{evidence_date}T00:00:00Z",
        "evidence_date": evidence_date,
        "staleness": {
            "max_age_days": 30,
            "status": "fresh",
            "source": "generated_at",
        },
        "source_artifacts": {
            "release_maturity_dashboard": _rel(OUTPUT),
            "production_docs_verification": "docs/source/_static/astro_starlight/production_docs_verification.json",
            "registry_submission_inventory": _rel(SOURCE_ARTIFACTS["registry_state"]),
            "binding_conformance_inventory": _rel(SOURCE_ARTIFACTS["binding_conformance"]),
        },
        "pages": pages,
    }


def _dashboard_page(slug_prefix: str = "") -> str:
    slug_line = f"slug: {slug_prefix}operations/release-maturity\n" if slug_prefix else ""
    return f"""---
title: Release Maturity Dashboard
description: Evidence-backed release, registry, binding, and Rust ownership status.
{slug_line}---

# Release Maturity Dashboard

This page is generated from machine-readable evidence. It is a status surface,
not a release announcement.

Source dashboard:

- `docs/source/_static/astro_starlight/release_maturity_dashboard.json`

Source artifacts:

- `docs/source/_static/release_readiness_contract.json`
- `docs/source/_static/rust_full_ownership_gate.json`
- `docs/source/_static/registry_submission_inventory.json`
- `docs/source/_static/binding_conformance_inventory.json`

Status summary:

- Release readiness: release-candidate evidence contract is defined.
- Rust ownership: full Rust ownership is not claimed.
- Registry state: not all external registries are accepted.
- Binding conformance: supported bindings are documented against the kernel
  contract.

Guardrails:

- Do not claim all registries accepted until every external registry artifact
  shows accepted or published evidence.
- Do not claim full Rust ownership until the Rust ownership gate allows that
  claim.
"""


def _policy_page(title: str, slug: str, summary: str, slug_prefix: str = "") -> str:
    slug_line = f"slug: {slug_prefix}maintainers/{slug}\n" if slug_prefix else ""
    return f"""---
title: {title}
description: Evidence-backed {title.lower()} for Innovate maintainers.
{slug_line}---

# {title}

{summary}

Evidence:

- `docs/source/_static/astro_starlight/observability_maintenance.json`
- `docs/source/_static/astro_starlight/release_maturity_dashboard.json`
- `docs/source/_static/astro_starlight/production_docs_verification.json`

Operational rules:

- Keep user-facing claims aligned with the machine-readable evidence artifacts.
- Treat external registry or deployment state as pending unless the relevant
  artifact shows accepted, published, or passed evidence.
- Refresh the evidence artifacts before a public release announcement.
"""


def write_outputs(dashboard: dict[str, Any]) -> None:
    """Write dashboard JSON and Starlight pages."""
    observability = build_observability(dashboard)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(dashboard, indent=2, sort_keys=True) + "\n")
    OBSERVABILITY_OUTPUT.write_text(json.dumps(observability, indent=2, sort_keys=True) + "\n")

    pages = {
        ASTRO_DOCS / "operations/release-maturity.md": _dashboard_page(),
        ASTRO_DOCS / "latest/operations/release-maturity.md": _dashboard_page("latest/"),
    }
    policy_pages = {
        "package-health": (
            "Package Health",
            "Package health summarizes package, binding, registry, and docs evidence before release.",
        ),
        "compatibility": (
            "Compatibility Policy",
            "Compatibility policy defines how supported languages and kernel schema commitments are presented.",
        ),
        "deprecation": (
            "Deprecation Policy",
            "Deprecation policy keeps removals and migrations tied to release readiness evidence.",
        ),
        "support": (
            "Support Policy",
            "Support policy names the maintained docs, registry, and issue-reporting surfaces.",
        ),
        "maintenance": (
            "Maintenance Policy",
            "Maintenance policy defines the evidence refresh and release readiness cadence.",
        ),
    }
    for slug, (title, summary) in policy_pages.items():
        pages[ASTRO_DOCS / f"maintainers/{slug}.md"] = _policy_page(title, slug, summary)
        pages[ASTRO_DOCS / f"latest/maintainers/{slug}.md"] = _policy_page(title, slug, summary, "latest/")
    for path, content in pages.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print dashboard JSON to stdout.")
    args = parser.parse_args()

    dashboard = build_dashboard()
    write_outputs(dashboard)
    if args.json:
        print(json.dumps(dashboard, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
