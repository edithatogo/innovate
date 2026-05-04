# R Package Manual and Vignette Quality Notes

## Standards Audit

- R Core documents `R CMD Rd2pdf` as the standard command for generating PDF
  output from Rd files and package source directories.
- R Core documents `R CMD build` as rebuilding package vignettes by default
  unless `--no-build-vignettes` or package metadata disables them.
- R Core documents `R CMD check` as testing vignette code and verifying that
  vignette production succeeded.
- `r-lib/actions` v2 provides maintained GitHub Actions for R setup,
  dependency setup, package checking, and TinyTeX setup.

The track keeps handwritten Rd documentation. The R binding has a deliberately
small exported surface, and the current static alias gate is enough to keep
handwritten docs aligned without introducing roxygen2 generation and its
additional dependency surface.

## Six-Agent Ownership

- Agent A owned R reference docs in `bindings/r/man/`.
- Agent B owned vignette sources in `bindings/r/vignettes/`.
- Agent C owned workflow manual-artifact gates in `.github/workflows/`.
- Agent D owned R metadata and CRAN policy files in `bindings/r/DESCRIPTION`,
  `bindings/r/.Rbuildignore`, and `bindings/r/cran-comments.md`.
- Agent E owned static quality gates in `tests/unit/test_r_package_pdf_manual.py`.
- Agent F owned user and release docs in `bindings/r/README.md` and
  `docs/source/`.

## Independent Validation Commands

- `uv run pytest tests/unit/test_r_package_pdf_manual.py`
- `uv run pytest tests/unit/test_binding_publication_ci.py`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `R CMD build bindings/r --no-manual`
- `R CMD Rd2pdf --no-preview --output=/tmp/innovate.R_manual_check.pdf bindings/r`
- `R CMD check --as-cran --no-manual innovate.R_0.5.0.tar.gz`
- `uv run python -m sphinx -b html docs/source /tmp/innovate-docs-r-manual-track`
