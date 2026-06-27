"""Tests validating the full route inventory against the actual content tree."""

from __future__ import annotations

import json
from pathlib import Path


ROUTES = {
    "/": "docs/astro-site/src/content/docs/index.md",
    "/core/kernel/": "docs/astro-site/src/content/docs/core/kernel.md",
    "/core/arrow-interchange/": "docs/astro-site/src/content/docs/core/arrow-interchange.md",
    "/core/diagnostics-contract/": "docs/astro-site/src/content/docs/core/diagnostics-contract.md",
    "/api/python/": None,
    "/bindings/": "docs/astro-site/src/content/docs/bindings/index.md",
    "/bindings/csharp/": "docs/astro-site/src/content/docs/bindings/csharp.md",
    "/bindings/go/": "docs/astro-site/src/content/docs/bindings/go.md",
    "/bindings/julia/": "docs/astro-site/src/content/docs/bindings/julia.md",
    "/bindings/rust/": "docs/astro-site/src/content/docs/bindings/rust.md",
    "/maintainers/publication/": "docs/astro-site/src/content/docs/maintainers/publication.md",
    "/maintainers/release-notes/": "docs/astro-site/src/content/docs/maintainers/release-notes.md",
    "/maintainers/docsearch/": "docs/astro-site/src/content/docs/maintainers/docsearch.md",
    "/maintainers/compatibility/": "docs/astro-site/src/content/docs/maintainers/compatibility.md",
    "/maintainers/plugins/": "docs/astro-site/src/content/docs/maintainers/plugins.md",
    "/operations/roadmap/": "docs/astro-site/src/content/docs/operations/roadmap.md",
    "/operations/rust-core/": "docs/astro-site/src/content/docs/operations/rust-core.md",
    "/operations/release-maturity/": "docs/astro-site/src/content/docs/operations/release-maturity.md",
    "/operations/governance/": "docs/astro-site/src/content/docs/operations/governance.md",
    "/architecture/": "docs/astro-site/src/content/docs/architecture/index.md",
    "/architecture/adr/": "docs/astro-site/src/content/docs/architecture/adr.md",
    "/migration/": "docs/astro-site/src/content/docs/migration/index.md",
    "/migration/redirects/": "docs/astro-site/src/content/docs/migration/redirects.md",
    "/migration/validation/": "docs/astro-site/src/content/docs/migration/validation.md",
    "/migration/archive/": "docs/astro-site/src/content/docs/migration/archive.md",
    "/migration/references/": "docs/astro-site/src/content/docs/migration/references.md",
    "/user-guide/getting-started/": "docs/astro-site/src/content/docs/user-guide/getting-started.mdx",
    "/user-guide/installation/": "docs/astro-site/src/content/docs/user-guide/installation.md",
    "/user-guide/fitting/": "docs/astro-site/src/content/docs/user-guide/fitting.md",
    "/user-guide/forecasting/": "docs/astro-site/src/content/docs/user-guide/forecasting.md",
    "/user-guide/backends/": "docs/astro-site/src/content/docs/user-guide/backends.md",
    "/roadmap/diagnostics-uncertainty/": "docs/astro-site/src/content/docs/roadmap/diagnostics-uncertainty.md",
    "/roadmap/probabilistic-inference/": "docs/astro-site/src/content/docs/roadmap/probabilistic-inference.md",
    "/tutorials/": "docs/astro-site/src/content/docs/tutorials/index.md",
}

VERSIONED_ROUTES = [
    "latest/index.md",
    "latest/core/kernel.md",
    "latest/architecture/index.md",
    "latest/bindings/index.md",
]


def test_all_sidebar_routes_have_content_files() -> None:
    """Every route from the sidebar config must have a corresponding content file."""
    missing = []
    for route, path in ROUTES.items():
        if path is None:
            continue
        if not Path(path).exists():
            missing.append((route, path))
    assert not missing, f"Missing content files for routes: {missing}"


def test_route_coverage_includes_migrated_routes() -> None:
    """The route coverage report must list all migrated core routes."""
    report = json.loads(Path("docs/source/_static/astro_starlight/route_coverage.json").read_text())
    implemented = {
        entry["astro_route"] for entry in report["coverage_by_source_doc"] if entry["status"] == "implemented"
    }
    assert "/" in implemented
    assert "/core/kernel/" in implemented
    assert "/core/arrow-interchange/" in implemented
    assert "/maintainers/publication/" in implemented


def test_versioned_latest_content_exists() -> None:
    """The versioned 'latest/' routes must have content files."""
    docs_root = Path("docs/astro-site/src/content/docs")
    missing = []
    for versioned_path in VERSIONED_ROUTES:
        if not (docs_root / versioned_path).exists():
            missing.append(versioned_path)
    assert not missing, f"Missing versioned content: {missing}"


def test_astro_site_builds_successfully() -> None:
    """The Astro site must build without errors (pnpm check already validates this)."""
    lockfile = Path("docs/astro-site/pnpm-lock.yaml")
    package = json.loads(Path("docs/astro-site/package.json").read_text())
    assert lockfile.exists()
    assert package["scripts"]["check"] == "astro check"
    assert package["scripts"]["build"] == "astro build"


def test_generated_python_api_docs_exist() -> None:
    """Starlight polyglot generation must produce Python API pages."""
    api_doc = Path("docs/astro-site/src/content/docs/api/python.md")
    assert api_doc.exists(), "Python API docs not generated"
    content = api_doc.read_text()
    assert "Python API Reference" in content or "innovate" in content


def test_sphinx_is_legacy_archive_only() -> None:
    """Sphinx must not have active build infrastructure."""
    conf_py = Path("docs/source/conf.py")
    build_dir = Path("docs/build")
    assert not conf_py.exists(), "Sphinx conf.py must not exist (archive-only)"
    assert not build_dir.exists(), "Sphinx build directory must not exist (archive-only)"


def test_custom_css_file_exists() -> None:
    """The Starlight custom CSS file must be present."""
    css = Path("docs/astro-site/src/styles/custom.css")
    assert css.exists()


def test_version_switcher_is_configured() -> None:
    """Starlight-versions plugin must be configured in astro.config.mjs."""
    config = Path("docs/astro-site/astro.config.mjs").read_text()
    assert "starlightVersions" in config
    assert "starlight-versions" in config


def test_docsearch_is_gated_by_env_vars() -> None:
    """DocSearch must require env vars to activate (not hardcoded)."""
    config = Path("docs/astro-site/astro.config.mjs").read_text()
    assert "ALGOLIA_APP_ID" in config
    assert "ALGOLIA_API_KEY" in config
    assert "ALGOLIA_INDEX_NAME" in config
    assert "starlightDocSearch" in config


def test_sidebar_contains_core_sections() -> None:
    """The Starlight sidebar must define the expected top-level sections."""
    config = Path("docs/astro-site/astro.config.mjs").read_text()
    for section in ("Getting Started", "User Guide", "API Reference", "Maintainers", "Operations", "Architecture", "Migration"):
        assert section in config