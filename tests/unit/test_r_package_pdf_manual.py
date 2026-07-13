"""Static gates for R package manual and vignette publication quality."""

from __future__ import annotations

import re
from pathlib import Path

R_BINDING_ROOT = Path("bindings/r")


def _exports() -> list[str]:
    namespace = (R_BINDING_ROOT / "NAMESPACE").read_text()
    return re.findall(r"^export\(([^)]+)\)$", namespace, flags=re.MULTILINE)


def _rd_aliases() -> set[str]:
    aliases: set[str] = set()
    for rd_file in (R_BINDING_ROOT / "man").glob("*.Rd"):
        aliases.update(re.findall(r"\\alias\{([^}]+)\}", rd_file.read_text()))
    return aliases


def test_exported_r_symbols_have_rd_aliases() -> None:
    """Every exported R helper should appear in Rd aliases."""
    assert set(_exports()) <= _rd_aliases()


def test_r_package_has_package_level_manual_page() -> None:
    """The R package should expose package-level context in the PDF manual."""
    package_page = R_BINDING_ROOT / "man" / "innovate.R-package.Rd"

    assert package_page.exists()
    text = package_page.read_text()
    assert "\\docType{package}" in text
    assert "\\alias{innovate.R-package}" in text
    assert "\\alias{innovate.R}" in text


def test_r_vignette_metadata_and_source_are_present() -> None:
    """Vignette sources and DESCRIPTION metadata should build under R CMD build."""
    description = (R_BINDING_ROOT / "DESCRIPTION").read_text()
    vignette_sources = list((R_BINDING_ROOT / "vignettes").glob("*.Rmd"))

    assert "VignetteBuilder: knitr" in description
    assert "knitr" in description
    assert "rmarkdown" in description
    assert vignette_sources
    vignette = vignette_sources[0].read_text()
    assert "%\\VignetteEngine{knitr::rmarkdown}" in vignette
    assert re.search(r"eval\s*=\s*FALSE", vignette)


def test_r_manual_policy_is_documented_for_users_and_releases() -> None:
    """Manual generation commands and artifact policy should be explicit."""
    readme = (R_BINDING_ROOT / "README.md").read_text()
    publication_docs = Path("docs/astro-site/src/content/docs/maintainers/publication.md").read_text()
    cran_comments = (R_BINDING_ROOT / "cran-comments.md").read_text()

    for text in (readme, publication_docs, cran_comments):
        assert "R CMD Rd2pdf" in text
        assert "PDF manual" in text

    assert "generated" in readme.lower()
    assert "artifact" in publication_docs.lower()
    assert "not committed" in readme.lower() or "do not commit" in readme.lower()


def test_r_workflows_build_and_upload_manual_pdf() -> None:
    """CI and publication workflows should block on R manual generation."""
    ci = Path(".github/workflows/ci.yml").read_text()
    publish = Path(".github/workflows/bindings-publish.yml").read_text()

    for workflow in (ci, publish):
        for action in (
            "r-lib/actions/setup-r",
            "r-lib/actions/setup-tinytex",
            "r-lib/actions/setup-pandoc",
            "actions/upload-artifact",
        ):
            assert re.search(rf"{re.escape(action)}@[0-9a-f]{{40}}\b", workflow)
        assert 'Sys.getenv("RSPM", "https://cloud.r-project.org")' in workflow
        assert "requireNamespace" in workflow
        assert "R CMD Rd2pdf" in workflow
        assert "manual.pdf" in workflow

    assert "R CMD check --as-cran --no-manual innovate.R_*.tar.gz" in ci
    assert "R CMD check --as-cran --no-manual innovate.R_*.tar.gz" in publish
