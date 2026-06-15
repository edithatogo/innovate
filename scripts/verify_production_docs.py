"""Verify production documentation evidence for the Astro/Starlight site."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ASTRO_ROOT = ROOT / "docs/astro-site"
ASTRO_DIST = ASTRO_ROOT / "dist"
EVIDENCE_PATH = ROOT / "docs/source/_static/astro_starlight/production_docs_verification.json"
ROUTE_COVERAGE = ROOT / "docs/source/_static/astro_starlight/route_coverage.json"
REDIRECT_INVENTORY = ROOT / "docs/source/_static/astro_starlight/redirect_inventory.json"
CUTOVER_VERIFICATION = ROOT / "docs/source/_static/astro_starlight/cutover_verification.json"
LINK_VALIDATION = ROOT / "docs/source/_static/astro_starlight/link_validation_report.json"
ASTRO_CONFIG = ASTRO_ROOT / "astro.config.mjs"
DOCS_WORKFLOW = ROOT / ".github/workflows/docs.yml"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _file_status(path: Path) -> str:
    return "passed" if path.exists() and path.stat().st_size > 0 else "failed"


def _route_file(route: str) -> Path:
    route = route.strip("/")
    if not route:
        return ASTRO_DIST / "index.html"
    return ASTRO_DIST / route / "index.html"


def _route_status(required_routes: list[str]) -> str:
    return "passed" if all(_route_file(route).exists() for route in required_routes) else "failed"


def build_evidence() -> dict[str, Any]:
    """Build the production docs verification evidence payload."""
    route_coverage = _read_json(ROUTE_COVERAGE)
    cutover = _read_json(CUTOVER_VERIFICATION)
    link_validation = _read_json(LINK_VALIDATION)
    astro_config = ASTRO_CONFIG.read_text()
    docs_workflow = DOCS_WORKFLOW.read_text()

    required_sitemaps = [
        ASTRO_DIST / "sitemap-index.xml",
        ASTRO_DIST / "sitemap-0.xml",
    ]
    required_versioned_routes = ["/latest/", "/latest/api/python/"]
    required_api_routes = ["/api/python/", "/latest/api/python/"]

    docsearch_env_vars = ["ALGOLIA_APP_ID", "ALGOLIA_API_KEY", "ALGOLIA_INDEX_NAME"]
    docsearch_has_safe_gate = all(env_var in astro_config for env_var in docsearch_env_vars)
    docsearch_has_spread = "...docSearchPlugins" in astro_config

    checks = [
        {
            "id": "route_coverage",
            "status": "passed"
            if route_coverage["counts"]["planned"] == 0
            and route_coverage["counts"]["implemented"] == route_coverage["counts"]["total"]
            and link_validation["counts"]["broken_links"] == 0
            else "failed",
            "source": _rel(ROUTE_COVERAGE),
            "evidence": {
                "implemented": route_coverage["counts"]["implemented"],
                "planned": route_coverage["counts"]["planned"],
                "broken_links": link_validation["counts"]["broken_links"],
            },
        },
        {
            "id": "redirect_inventory",
            "status": "passed"
            if cutover["ready_for_cutover"] is True and cutover["counts"]["redirect_only"] == 0
            else "failed",
            "source": _rel(CUTOVER_VERIFICATION),
            "evidence": {
                "redirect_entries": cutover["counts"]["redirect_entries"],
                "content_only": cutover["counts"]["content_only"],
                "redirect_only": cutover["counts"]["redirect_only"],
            },
        },
        {
            "id": "sitemap",
            "status": "passed" if all(_file_status(path) == "passed" for path in required_sitemaps) else "failed",
            "source": _rel(ASTRO_DIST),
            "evidence": {"required_files": [_rel(path) for path in required_sitemaps]},
        },
        {
            "id": "search_configuration",
            "status": "ci_safe" if docsearch_has_safe_gate and docsearch_has_spread else "failed",
            "source": _rel(ASTRO_CONFIG),
            "evidence": {
                "provider": "algolia-docsearch",
                "fallback_without_credentials": docsearch_has_safe_gate and docsearch_has_spread,
                "required_environment": docsearch_env_vars,
            },
        },
        {
            "id": "versioned_docs",
            "status": _route_status(required_versioned_routes),
            "source": _rel(ASTRO_DIST),
            "evidence": {"required_routes": required_versioned_routes},
        },
        {
            "id": "api_generation",
            "status": _route_status(required_api_routes),
            "source": _rel(ASTRO_DIST),
            "evidence": {
                "generator": "starlight-polyglot",
                "required_routes": required_api_routes,
            },
        },
        {
            "id": "ci_workflow",
            "status": "passed"
            if "Verify production documentation contract" in docs_workflow
            and "verify_production_docs.py" in docs_workflow
            else "failed",
            "source": _rel(DOCS_WORKFLOW),
            "evidence": {
                "workflow": "Deploy Documentation",
                "required_step": "Verify production documentation contract",
            },
        },
    ]

    overall_status = "passed" if all(check["status"] in {"passed", "ci_safe"} for check in checks) else "failed"
    return {
        "schema_version": 1,
        "generated_by_track": "production_docs_observability_20260614",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "evidence_date": date.today().isoformat(),
        "overall_status": overall_status,
        "staleness": {
            "max_age_days": 30,
            "status": "fresh",
            "source": "generated_at",
        },
        "generated_from": {
            "route_coverage": _rel(ROUTE_COVERAGE),
            "redirect_inventory": _rel(REDIRECT_INVENTORY),
            "cutover_verification": _rel(CUTOVER_VERIFICATION),
            "link_validation": _rel(LINK_VALIDATION),
            "astro_config": _rel(ASTRO_CONFIG),
            "docs_workflow": _rel(DOCS_WORKFLOW),
            "dist": _rel(ASTRO_DIST),
        },
        "checks": checks,
        "commands": [
            {
                "command": "python scripts/verify_production_docs.py --json",
                "status": "passed" if overall_status == "passed" else "failed",
            },
            {
                "command": "pnpm build && python ../../scripts/verify_production_docs.py --json",
                "status": "ci_wired",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print evidence JSON to stdout.")
    parser.add_argument(
        "--output",
        type=Path,
        default=EVIDENCE_PATH,
        help="Evidence file to write.",
    )
    args = parser.parse_args()

    evidence = build_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
