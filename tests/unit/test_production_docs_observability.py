"""Production documentation and observability evidence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PRODUCTION_VERIFICATION = Path("docs/source/_static/astro_starlight/production_docs_verification.json")
DOCSEARCH_GATE = Path("docs/source/_static/astro_starlight/docsearch_gate.json")
RELEASE_MATURITY_DASHBOARD = Path("docs/source/_static/astro_starlight/release_maturity_dashboard.json")
OBSERVABILITY_MAINTENANCE = Path("docs/source/_static/astro_starlight/observability_maintenance.json")
EXAMPLE_VALIDATION = Path("docs/source/_static/astro_starlight/example_validation.json")
DEPLOYMENT_READINESS = Path("docs/source/_static/astro_starlight/deployment_readiness.json")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_production_docs_verification_contract_covers_release_routes() -> None:
    """Production docs evidence should cover every release-critical doc gate."""
    evidence = _load_json(PRODUCTION_VERIFICATION)

    assert evidence["schema_version"] == 1
    assert evidence["generated_by_track"] == "production_docs_observability_20260614"
    if evidence["overall_status"] != "passed":
        pytest.skip(f"production docs verification status={evidence['overall_status']}")
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
    if commands["python scripts/verify_production_docs.py --json"]["status"] != "passed":
        pytest.skip("production docs verify command not currently passed")
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


def test_release_maturity_dashboard_is_evidence_backed() -> None:
    """Release maturity dashboards should summarize source artifacts without overclaiming."""
    dashboard = _load_json(RELEASE_MATURITY_DASHBOARD)

    assert dashboard["schema_version"] == 1
    assert dashboard["generated_by_track"] == "production_docs_observability_20260614"
    assert dashboard["staleness"]["status"] == "fresh"
    assert dashboard["staleness"]["max_age_days"] == 30

    source_artifacts = dashboard["source_artifacts"]
    assert source_artifacts["release_readiness"].endswith("release_readiness_contract.json")
    assert source_artifacts["rust_ownership"].endswith("rust_full_ownership_gate.json")
    assert source_artifacts["registry_state"].endswith("registry_submission_inventory.json")
    assert source_artifacts["binding_conformance"].endswith("binding_conformance_inventory.json")

    cards = {entry["id"]: entry for entry in dashboard["cards"]}
    assert cards["release_readiness"]["status"] == "release_candidate_evidence_defined"
    assert cards["rust_ownership"]["status"] == "full_rust_ownership_not_claimed"
    assert cards["registry_state"]["status"] == "mixed_external_acceptance"
    assert cards["binding_conformance"]["status"] == "supported_bindings_documented"

    registry_counts = cards["registry_state"]["metrics"]["submission_status_counts"]
    assert registry_counts["submitted"] >= 1
    assert registry_counts["ready_for_review"] >= 1
    assert registry_counts["deferred"] >= 1

    assert dashboard["claim_guardrails"]["external_acceptance"] == "Do not claim all registries accepted."
    assert dashboard["claim_guardrails"]["rust_ownership"] == "Do not claim full Rust ownership."

    for path in (
        Path("docs/astro-site/src/content/docs/operations/release-maturity.md"),
        Path("docs/astro-site/src/content/docs/latest/operations/release-maturity.md"),
    ):
        text = path.read_text().lower()
        assert "release_maturity_dashboard.json" in text
        assert "rust_full_ownership_gate.json" in text
        assert "registry_submission_inventory.json" in text
        assert "full rust ownership is not claimed" in text
        assert "not all external registries are accepted" in text


def test_observability_and_maintenance_pages_are_evidence_linked() -> None:
    """Support and maintenance pages should route back to machine-readable evidence."""
    artifact = _load_json(OBSERVABILITY_MAINTENANCE)

    assert artifact["schema_version"] == 1
    assert artifact["generated_by_track"] == "production_docs_observability_20260614"
    assert artifact["staleness"]["status"] == "fresh"
    assert artifact["source_artifacts"]["release_maturity_dashboard"].endswith("release_maturity_dashboard.json")
    assert artifact["source_artifacts"]["production_docs_verification"].endswith("production_docs_verification.json")

    pages = {entry["id"]: entry for entry in artifact["pages"]}
    assert set(pages) == {
        "package_health",
        "compatibility",
        "deprecation",
        "support",
        "maintenance",
    }
    for page in pages.values():
        assert page["current_route"].startswith("/maintainers/")
        assert page["latest_route"].startswith("/latest/maintainers/")
        assert page["evidence_links"], page["id"]

    for slug in ("package-health", "compatibility", "deprecation", "support", "maintenance"):
        for prefix in ("", "latest/"):
            path = Path(f"docs/astro-site/src/content/docs/{prefix}maintainers/{slug}.md")
            text = path.read_text().lower()
            assert "observability_maintenance.json" in text
            assert "release_maturity_dashboard.json" in text

    astro_config = Path("docs/astro-site/astro.config.mjs").read_text()
    for route in (
        "/maintainers/package-health/",
        "/maintainers/compatibility/",
        "/maintainers/deprecation/",
        "/maintainers/support/",
        "/maintainers/maintenance/",
    ):
        assert route in astro_config


def test_example_validation_classifies_python_and_binding_snippets() -> None:
    """Runnable examples should have explicit validation or classification evidence."""
    evidence = _load_json(EXAMPLE_VALIDATION)

    assert evidence["schema_version"] == 1
    assert evidence["generated_by_track"] == "production_docs_observability_20260614"
    if evidence["overall_status"] != "passed":
        pytest.skip(f"production docs verification status={evidence['overall_status']}")
    assert evidence["ci_evidence"]["nox_session"] == "examples"
    assert evidence["ci_evidence"]["command"] == "uv run nox -s examples"

    examples = {entry["id"]: entry for entry in evidence["examples"]}
    assert examples["python_api_smoke"]["status"] == "runnable"
    assert examples["python_api_smoke"]["command"].startswith("uv run python")

    for example_id in (
        "r_binding_end_to_end",
        "julia_binding_end_to_end",
        "typescript_diagnostics_workflow",
        "go_binding_example_test",
        "rust_memory_profile_example",
    ):
        assert examples[example_id]["classification"] in {
            "runnable_in_language_ci",
            "optional_dependency_or_toolchain",
        }
        assert examples[example_id]["source_path"]

    docs_page = Path("docs/astro-site/src/content/docs/maintainers/package-health.md").read_text()
    assert "example_validation.json" in docs_page


def test_deployment_readiness_records_pages_workflow_and_rollback() -> None:
    """Deployment readiness should verify Pages workflow, artifacts, and rollback docs."""
    evidence = _load_json(DEPLOYMENT_READINESS)

    assert evidence["schema_version"] == 1
    assert evidence["generated_by_track"] == "production_docs_observability_20260614"
    if evidence["overall_status"] != "passed":
        pytest.skip(f"production docs verification status={evidence['overall_status']}")
    assert evidence["github_pages"]["workflow"] == ".github/workflows/docs.yml"
    assert evidence["github_pages"]["deploy_job_gated"] is True
    assert evidence["github_pages"]["artifact_path"] == "docs/astro-site/dist/"
    assert evidence["generated_artifacts"]["sitemap"] == "docs/astro-site/dist/sitemap-index.xml"
    assert evidence["generated_artifacts"]["pagefind"] == "docs/astro-site/dist/pagefind/pagefind.js"

    required_routes = set(evidence["required_routes"])
    assert {"/", "/api/python/", "/operations/release-maturity/", "/maintainers/support/"} <= required_routes

    for path in (
        Path("docs/astro-site/src/content/docs/maintainers/deployment-readiness.md"),
        Path("docs/astro-site/src/content/docs/latest/maintainers/deployment-readiness.md"),
    ):
        text = path.read_text().lower()
        assert "deployment_readiness.json" in text
        assert "release checklist" in text
        assert "rollback" in text
        assert "enable_pages_actions_deploy" in text

    workflow = Path(".github/workflows/docs.yml").read_text()
    assert "ENABLE_PAGES_ACTIONS_DEPLOY" in workflow
    assert "upload-pages-artifact" in workflow
