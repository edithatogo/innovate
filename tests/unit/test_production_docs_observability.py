"""Production documentation and observability evidence tests."""

from __future__ import annotations

import json
from pathlib import Path

PRODUCTION_VERIFICATION = Path("docs/source/_static/astro_starlight/production_docs_verification.json")
DOCSEARCH_GATE = Path("docs/source/_static/astro_starlight/docsearch_gate.json")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_production_docs_verification_contract_covers_release_routes() -> None:
    """Production docs evidence should cover every release-critical doc gate."""
    evidence = _load_json(PRODUCTION_VERIFICATION)

    assert evidence["schema_version"] == 1
    assert evidence["generated_by_track"] == "production_docs_observability_20260614"
    assert evidence["overall_status"] == "passed"
    assert evidence["staleness"]["max_age_days"] == 30
    assert evidence["staleness"]["status"] == "fresh"

    checks = {entry["id"]: entry for entry in evidence["checks"]}
    assert set(checks) >= {
        "route_coverage",
        "redirect_inventory",
        "sitemap",
        "search_configuration",
        "versioned_docs",
        "api_generation",
        "ci_workflow",
    }

    for check_id, check in checks.items():
        assert check["status"] in {"passed", "ci_safe"}, check_id
        assert check["evidence"], check_id
        assert check["source"], check_id

    assert checks["sitemap"]["evidence"]["required_files"] == [
        "docs/astro-site/dist/sitemap-index.xml",
        "docs/astro-site/dist/sitemap-0.xml",
    ]
    assert checks["search_configuration"]["evidence"]["fallback_without_credentials"] is True
    assert checks["versioned_docs"]["evidence"]["required_routes"] == [
        "/latest/",
        "/latest/api/python/",
    ]


def test_production_docs_verification_commands_are_documented_and_ci_wired() -> None:
    """The production verification contract should be runnable locally and in CI."""
    evidence = _load_json(PRODUCTION_VERIFICATION)
    commands = {entry["command"]: entry for entry in evidence["commands"]}

    assert "python scripts/verify_production_docs.py --json" in commands
    assert commands["python scripts/verify_production_docs.py --json"]["status"] == "passed"
    assert commands["pnpm build && python ../../scripts/verify_production_docs.py --json"]["status"] == "ci_wired"

    docs_readme = Path("docs/astro-site/README.md").read_text()
    docs_workflow = Path(".github/workflows/docs.yml").read_text()
    noxfile = Path("noxfile.py").read_text()

    assert "python ../../scripts/verify_production_docs.py --json" in docs_readme
    assert "Verify production documentation contract" in docs_workflow
    assert "verify_production_docs.py" in docs_workflow
    assert "def production_docs" in noxfile


def test_docsearch_gate_documents_safe_secret_boundaries() -> None:
    """DocSearch evidence should separate local fallback from production enablement."""
    gate = _load_json(DOCSEARCH_GATE)

    assert gate["schema_version"] == 1
    assert gate["provider"] == "algolia-docsearch"
    assert gate["current_local_status"] == "disabled_without_credentials"
    assert gate["production_status"] == "external_credentials_required"
    assert gate["credential_policy"]["hard_code_credentials"] is False
    assert gate["credential_policy"]["required_environment"] == [
        "ALGOLIA_APP_ID",
        "ALGOLIA_API_KEY",
        "ALGOLIA_INDEX_NAME",
    ]

    modes = {entry["status"]: entry for entry in gate["modes"]}
    assert modes["enabled"]["evidence_fields"] == ["app_id_present", "api_key_present", "index_name_present"]
    assert modes["disabled_without_credentials"]["ci_safe"] is True
    assert modes["external_credentials_required"]["owner"] == "deployment_environment"

    production = _load_json(PRODUCTION_VERIFICATION)
    search_check = {entry["id"]: entry for entry in production["checks"]}["search_configuration"]
    assert search_check["evidence"]["docsearch_gate"] == str(DOCSEARCH_GATE)

    for path in (
        Path("docs/astro-site/src/content/docs/maintainers/docsearch.md"),
        Path("docs/astro-site/src/content/docs/latest/maintainers/docsearch.md"),
    ):
        text = path.read_text().lower()
        assert "algolia_app_id" in text
        assert "disabled_without_credentials" in text
        assert "external_credentials_required" in text
        assert "do not hard-code" in text
