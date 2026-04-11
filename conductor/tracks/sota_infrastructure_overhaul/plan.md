# Implementation Plan: SOTA Infrastructure Overhaul

## Phase 1: Dependency Migration to uv

- [x] Task: Analyze current dependency structure `f61e0b9`
    - [x] Review pyproject.toml dependencies and optional-dependencies `f61e0b9`
    - [x] Review requirements.txt for any missing dependencies `f61e0b9`
    - [x] Identify version constraints that need updating `f61e0b9`
- [x] Task: Rewrite pyproject.toml for uv
    - [x] Convert to uv-compatible format with proper dependency groups `dd9c657`
    - [x] Add uv-specific settings (override files, constraint files) `dd9c657`
    - [x] Pin Python minimum to 3.10 (3.8/3.9 are EOL) `dd9c657`
    - [x] Organize dependencies with clear comments `dd9c657`
    - [x] Add tool configurations for all SOTA tools `dd9c657`
- [x] Task: Generate uv.lock and verify `dd9c657`
    - [x] Run `uv lock` to generate lockfile `dd9c657`
    - [x] Run `uv sync` to verify all dependencies resolve correctly `dd9c657`
    - [x] Verify `uv run pytest` works with uv-managed environment `dd9c657`
- [x] Task: Update all documentation references `dd9c657`
    - [x] Update README.md installation instructions for uv `dd9c657`
    - [x] Update CONTRIBUTING.md development setup for uv `dd9c657`
    - [x] Update Dockerfile to use uv `dd9c657`
- [x] Task: Conductor - Automated Review 'Phase 1' (Protocol in workflow.md) `dd9c657`

## Phase 2: Ruff Consolidation (Replace Black, isort, flake8, Pylint, vulture, unimport)

- [x] Task: Configure Ruff as the single linting/formatting tool `1b6f694`
    - [x] Update `[tool.ruff]` section in pyproject.toml with comprehensive rules `1b6f694`
    - [x] Enable: F (pyflakes), E/W (pycodestyle), I (isort), B (bugbear), SIM (simplify), UP (pyupgrade), RUF (ruff-specific), C90 (mccabe), N (naming), D (docstyle) `1b6f694`
    - [x] Configure Ruff format to replace Black `1b6f694`
    - [x] Add per-file ignores for legacy code that can't be immediately fixed `1b6f694`
    - [x] Configure Ruff to detect unused imports/variables (replaces vulture + unimport) `1b6f694`
- [x] Task: Remove legacy linting tools `1b6f694`
    - [x] Remove Black, isort, flake8, Pylint, vulture, unimport from dev dependencies `1b6f694`
    - [x] Remove their configurations from pyproject.toml `1b6f694`
    - [x] Update all CI workflow references to use Ruff `1b6f694`
- [x] Task: Run Ruff and fix violations `1b6f694`
    - [x] Run `uv run ruff check . --fix` to auto-fix what's possible `1b6f694`
    - [x] Manually review and fix remaining violations `1b6f694`
    - [x] Run `uv run ruff format .` to format all code `1b6f694`
    - [x] Verify no linting errors remain `1b6f694`
- [x] Task: Conductor - Automated Review 'Phase 2' (Protocol in workflow.md) `1b6f694`

## Phase 3: Type Checking with ty + mypy

- [x] Task: Configure ty as primary type checker `cbaec8c`
    - [x] Add `[tool.ty]` configuration to pyproject.toml `cbaec8c`
    - [x] Run `uv run ty check src/` to identify current type issues `cbaec8c`
    - [x] Fix critical type errors that block adoption `cbaec8c`
- [x] Task: Update mypy configuration `cbaec8c`
    - [x] Keep mypy as secondary check with stricter modes `cbaec8c`
    - [x] Update mypy config in pyproject.toml to align with ty `cbaec8c`
    - [x] Ensure both ty and mypy can run in CI without conflicts `cbaec8c`
- [x] Task: Remove pyright `cbaec8c`
    - [x] Remove pyright from dev dependencies `cbaec8c`
    - [x] Remove pyright CI job from workflows `cbaec8c`
- [x] Task: Conductor - Automated Review 'Phase 3' (Protocol in workflow.md) `cbaec8c`

## Phase 4: Pre-commit Hooks Modernization

- [x] Task: Rewrite .pre-commit-config.yaml `cbaec8c`
    - [x] Replace Black hook with `ruff-format` `cbaec8c`
    - [x] Replace isort hook with `ruff-check --select I` `cbaec8c`
    - [x] Replace flake8 hook with `ruff-check` `cbaec8c`
    - [x] Remove vulture and unimport hooks (covered by Ruff) `cbaec8c`
    - [x] Update mypy hook to latest version `cbaec8c`
    - [x] Keep codespell, nbstripout hooks `cbaec8c`
    - [x] Add: check-yaml, check-toml, check-merge-conflict, end-of-file-fixer, trailing-whitespace `cbaec8c`
    - [x] Add: actionlint for GitHub Actions validation `cbaec8c`
    - [x] Configure Ruff hooks to run with `--fix` on commit `cbaec8c`
- [x] Task: Test pre-commit hooks `cbaec8c`
    - [x] Run `uv run pre-commit run --all-files` `cbaec8c`
    - [x] Verify all hooks pass `cbaec8c`
    - [x] Fix any violations found `cbaec8c`
- [x] Task: Conductor - Automated Review 'Phase 4' (Protocol in workflow.md) `cbaec8c`

## Phase 5: CI/CD Consolidation

- [x] Task: Design unified CI workflow `cbaec8c`
    - [x] Merge ci.yml, python_ci.yml, lint.yml into single `.github/workflows/ci.yml` `cbaec8c`
    - [x] Structure as matrix jobs: test (python 3.10-3.13), lint, type-check, security `cbaec8c`
    - [x] Update all actions to latest versions (checkout@v5, setup-python@v5) `cbaec8c`
    - [x] Add Scalene profiling as an optional benchmark job `cbaec8c`
    - [x] Add mutation testing (mutmut) as a weekly scheduled job `cbaec8c`
    - [x] Add integration and e2e test markers to the matrix `cbaec8c`
- [x] Task: Implement consolidated CI workflow `cbaec8c`
    - [x] Write new ci.yml with all jobs `cbaec8c`
    - [x] Ensure coverage upload to Codecov works `cbaec8c`
    - [x] Ensure all quality checks run in parallel `cbaec8c`
    - [x] Add CI gate monitoring job that polls after pushes `cbaec8c`
- [x] Task: Add CI Gate Monitoring workflow `cbaec8c`
    - [x] Create `.github/workflows/ci-gate-monitor.yml` `cbaec8c`
    - [x] Trigger on workflow_run completion `cbaec8c`
    - [x] Check status of all workflows triggered by the same push `cbaec8c`
    - [x] Post comment on PR or push notification if any fail `cbaec8c`
    - [x] Auto-create issue for persistent CI failures `cbaec8c`
- [x] Task: Delete obsolete CI workflows `cbaec8c`
    - [x] Remove old ci.yml, python_ci.yml, lint.yml `cbaec8c`
- [x] Task: Conductor - Automated Review 'Phase 5' (Protocol in workflow.md) `cbaec8c`

## Phase 6: Replace Dependabot with Renovate

- [x] Task: Create Renovate configuration `cbaec8c`
    - [x] Create `.github/renovate.json` `cbaec8c`
    - [x] Test Renovate configuration `cbaec8c`
- [x] Task: Delete Dependabot `cbaec8c`
    - [x] Remove `.github/dependabot.yml` `cbaec8c`
- [x] Task: Install Renovate GitHub App
    - [x] Document: Install Renovate from https://github.com/apps/renovate
- [x] Task: Conductor - Automated Review 'Phase 6' (Protocol in workflow.md) `cbaec8c`

## Phase 7: Add SOTA Files and Infrastructure

- [x] Task: Create CITATION.cff `cbaec8c`
    - [x] Add authors, title, abstract, version, URL, license `cbaec8c`
    - [x] Include preferred citation format for academic papers `cbaec8c`
    - [x] Validate with cff-validator `cbaec8c`
- [x] Task: Create CODEOWNERS `cbaec8c`
    - [x] Set repository owners for code, docs, and CI `cbaec8c`
- [x] Task: Create SECURITY.md `cbaec8c`
    - [x] Define security reporting process `cbaec8c`
- [x] Task: Create .editorconfig `cbaec8c`
    - [x] Define consistent indentation, line endings, charset `cbaec8c`
- [x] Task: Add actionlint CI job `cbaec8c`
    - [x] Add actionlint to the lint job in CI workflow `cbaec8c`
- [x] Task: Add release-please workflow `cbaec8c`
    - [x] Create `.github/workflows/release-please.yml` `cbaec8c`
    - [x] Replace release-drafter.yml `cbaec8c`
- [x] Task: Add Scalene profiling setup `cbaec8c`
    - [x] Add Scalene to dev dependencies `cbaec8c`
    - [x] Add profiling CI job `cbaec8c`
- [x] Task: Conductor - Automated Review 'Phase 7' (Protocol in workflow.md) `cbaec8c`

## Phase 8: Test Structure Reorganization

- [~] Task: Reorganize test directory structure
    - [ ] Create `tests/unit/` directory
    - [ ] Create `tests/integration/` directory (if not exists)
    - [ ] Create `tests/e2e/` directory (if not exists)
    - [ ] Move existing model-specific tests to `tests/unit/`
    - [ ] Move cross-module tests to `tests/integration/`
    - [ ] Move end-to-end workflow tests to `tests/e2e/`
- [ ] Task: Update pytest configuration
    - [ ] Add pytest markers: `unit`, `integration`, `e2e`
    - [ ] Configure default to run all tests
    - [ ] Add `pytest -m unit` for fast feedback during development
    - [ ] Update testpaths in pyproject.toml
- [ ] Task: Update CI to use new test structure
    - [ ] Update CI workflow to run unit tests on every commit
    - [ ] Run integration tests on PRs and pushes to main
    - [ ] Run e2e tests on pushes to main only
- [ ] Task: Conductor - Automated Review 'Phase 8' (Protocol in workflow.md)

## Phase 9: Final Quality Gate and Push

- [ ] Task: Run complete quality gate verification
    - [ ] `uv run pytest` — all tests pass
    - [ ] `uv run pytest --cov=innovate --cov-report=xml` — coverage >80%
    - [ ] `uv run ruff check .` — no linting errors
    - [ ] `uv run ruff format --check .` — all code formatted
    - [ ] `uv run ty check src/` — type checking passes
    - [ ] `uv run bandit -r src/innovate` — security scan passes
    - [ ] `uv run pre-commit run --all-files` — all hooks pass
- [ ] Task: Push all changes to remote
    - [ ] Push to feature branch first
    - [ ] Monitor CI gate — address all failures iteratively
    - [ ] Once CI passes, merge to main
- [ ] Task: Update conductor documentation
    - [ ] Update tech-stack.md with all new tools
    - [ ] Update product-guidelines.md with new standards
    - [ ] Update workflow.md with new development commands
- [ ] Task: Final cleanup
    - [ ] Remove any temporary files
    - [ ] Verify all git notes are complete
    - [ ] Ensure plan.md is fully updated
- [ ] Task: Conductor - Automated Review 'Phase 9' (Protocol in workflow.md)
