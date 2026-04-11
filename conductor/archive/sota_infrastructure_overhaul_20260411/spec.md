# Track Specification: SOTA Infrastructure Overhaul

## Overview
This track modernizes the entire development infrastructure of the `innovate` library to achieve state-of-the-art code management, maximize automation, and minimize maintenance overhead.

## Objectives
1. Migrate from pip/setuptools to `uv` for blazing-fast dependency management
2. Replace Dependabot with Renovate for intelligent, grouped dependency updates
3. Consolidate all linting/formatting under Ruff (replaces Black, isort, flake8, Pylint, vulture, unimport)
4. Make `ty` the primary type checker, drop redundant pyright
5. Modernize pre-commit hooks for the Ruff + uv era
6. Consolidate CI/CD into a single, efficient pipeline with matrix jobs
7. Add automated CI gate monitoring after every push
8. Formalize unit/integration/e2e test structure
9. Add Scalene profiling for performance analysis
10. Add CITATION.cff for academic citation
11. Add release-please for automated conventional-commit-driven releases
12. Add additional SOTA files: CODEOWNERS, SECURITY.md, .editorconfig, actionlint

## Scope
### In Scope
- `pyproject.toml` — Complete rewrite for uv + Ruff + ty + all SOTA tools
- `.pre-commit-config.yaml` — Modernized hooks
- `.github/workflows/` — Consolidated CI, Renovate config, release-please
- `.github/renovate.json` — Renovate configuration
- `.github/dependabot.yml` — Delete (replaced by Renovate)
- `CITATION.cff` — New citation file
- `CODEOWNERS` — New file
- `SECURITY.md` — New file
- `.editorconfig` — New file
- `.github/actionlint.yml` or actionlint CI job
- Test directory reorganization: `tests/unit/`, `tests/integration/`, `tests/e2e/`
- `conductor/tech-stack.md` — Update with new tools
- `conductor/product-guidelines.md` — Update with new standards

### Out of Scope
- New feature development
- Model implementation changes
- Documentation content updates (beyond infrastructure-related docs)

## Acceptance Criteria
- `uv sync` installs all dependencies correctly
- `uv run pytest` passes all tests
- `uv run ruff check .` and `uv run ruff format --check .` pass
- `uv run ty check src/` passes
- Renovate is configured and replaces Dependabot
- CI runs as a single consolidated workflow with matrix jobs
- Pre-commit hooks run Ruff, mypy, codespell, and safety checks
- CITATION.cff is valid and renders correctly on GitHub
- All tests reorganized into unit/integration/e2e structure
- Scalene profiling script available for benchmarking
- CI gate monitoring workflow implemented
- release-please configured for automated versioning and changelogs

## Technical Approach
1. **Infrastructure Analysis**: Audit current setup files and identify all replacements needed
2. **Dependency Migration**: Rewrite pyproject.toml for uv, generate uv.lock
3. **Linting Consolidation**: Configure Ruff to replace all legacy tools
4. **CI Consolidation**: Merge ci.yml, python_ci.yml, lint.yml into single workflow
5. **Renovate Setup**: Create renovate.json, delete dependabot.yml
6. **SOTA Files**: Add CITATION.cff, CODEOWNERS, SECURITY.md, .editorconfig
7. **Test Reorganization**: Move tests into unit/integration/e2e structure
8. **Pre-commit Modernization**: Update all hooks
9. **Release Automation**: Add release-please workflow
10. **Quality Gate**: Run full test suite, linting, type checking, and CI verification
