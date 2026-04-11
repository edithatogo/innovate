# Implementation Plan: SOTA Infrastructure Overhaul

## Phase 1: Dependency Migration to uv

- [ ] Task: Analyze current dependency structure
    - [ ] Review pyproject.toml dependencies and optional-dependencies
    - [ ] Review requirements.txt for any missing dependencies
    - [ ] Identify version constraints that need updating
- [ ] Task: Rewrite pyproject.toml for uv
    - [ ] Convert to uv-compatible format with proper dependency groups
    - [ ] Add uv-specific settings (override files, constraint files)
    - [ ] Pin Python minimum to 3.10 (3.8/3.9 are EOL)
    - [ ] Organize dependencies with clear comments
    - [ ] Add tool configurations for all SOTA tools
- [ ] Task: Generate uv.lock and verify
    - [ ] Run `uv lock` to generate lockfile
    - [ ] Run `uv sync` to verify all dependencies resolve correctly
    - [ ] Verify `uv run pytest` works with uv-managed environment
- [ ] Task: Update all documentation references
    - [ ] Update README.md installation instructions for uv
    - [ ] Update CONTRIBUTING.md development setup for uv
    - [ ] Update Dockerfile to use uv
- [ ] Task: Conductor - Automated Review 'Phase 1' (Protocol in workflow.md)

## Phase 2: Ruff Consolidation (Replace Black, isort, flake8, Pylint, vulture, unimport)

- [ ] Task: Configure Ruff as the single linting/formatting tool
    - [ ] Update `[tool.ruff]` section in pyproject.toml with comprehensive rules
    - [ ] Enable: F (pyflakes), E/W (pycodestyle), I (isort), B (bugbear), SIM (simplify), UP (pyupgrade), RUF (ruff-specific), C90 (mccabe), N (naming), D (docstyle)
    - [ ] Configure Ruff format to replace Black
    - [ ] Add per-file ignores for legacy code that can't be immediately fixed
    - [ ] Configure Ruff to detect unused imports/variables (replaces vulture + unimport)
- [ ] Task: Remove legacy linting tools
    - [ ] Remove Black, isort, flake8, Pylint, vulture, unimport from dev dependencies
    - [ ] Remove their configurations from pyproject.toml
    - [ ] Update all CI workflow references to use Ruff
- [ ] Task: Run Ruff and fix violations
    - [ ] Run `uv run ruff check . --fix` to auto-fix what's possible
    - [ ] Manually review and fix remaining violations
    - [ ] Run `uv run ruff format .` to format all code
    - [ ] Verify no linting errors remain
- [ ] Task: Conductor - Automated Review 'Phase 2' (Protocol in workflow.md)

## Phase 3: Type Checking with ty + mypy

- [ ] Task: Configure ty as primary type checker
    - [ ] Add `[tool.ty]` configuration to pyproject.toml
    - [ ] Run `uv run ty check src/` to identify current type issues
    - [ ] Fix critical type errors that block adoption
- [ ] Task: Update mypy configuration
    - [ ] Keep mypy as secondary check with stricter modes
    - [ ] Update mypy config in pyproject.toml to align with ty
    - [ ] Ensure both ty and mypy can run in CI without conflicts
- [ ] Task: Remove pyright
    - [ ] Remove pyright from dev dependencies
    - [ ] Remove pyright CI job from workflows
- [ ] Task: Conductor - Automated Review 'Phase 3' (Protocol in workflow.md)

## Phase 4: Pre-commit Hooks Modernization

- [ ] Task: Rewrite .pre-commit-config.yaml
    - [ ] Replace Black hook with `ruff-format`
    - [ ] Replace isort hook with `ruff-check --select I`
    - [ ] Replace flake8 hook with `ruff-check`
    - [ ] Remove vulture and unimport hooks (covered by Ruff)
    - [ ] Update mypy hook to latest version
    - [ ] Keep codespell, nbstripout hooks
    - [ ] Add: check-yaml, check-toml, check-merge-conflict, end-of-file-fixer, trailing-whitespace
    - [ ] Add: actionlint for GitHub Actions validation
    - [ ] Configure Ruff hooks to run with `--fix` on commit
- [ ] Task: Test pre-commit hooks
    - [ ] Run `uv run pre-commit run --all-files`
    - [ ] Verify all hooks pass
    - [ ] Fix any violations found
- [ ] Task: Conductor - Automated Review 'Phase 4' (Protocol in workflow.md)

## Phase 5: CI/CD Consolidation

- [ ] Task: Design unified CI workflow
    - [ ] Merge ci.yml, python_ci.yml, lint.yml into single `.github/workflows/ci.yml`
    - [ ] Structure as matrix jobs: test (python 3.10-3.13), lint, type-check, security
    - [ ] Update all actions to latest versions (checkout@v5, setup-python@v5)
    - [ ] Add Scalene profiling as an optional benchmark job
    - [ ] Add mutation testing (mutmut) as a weekly scheduled job
    - [ ] Add integration and e2e test markers to the matrix
- [ ] Task: Implement consolidated CI workflow
    - [ ] Write new ci.yml with all jobs
    - [ ] Ensure coverage upload to Codecov works
    - [ ] Ensure all quality checks run in parallel
    - [ ] Add CI gate monitoring job that polls after pushes
- [ ] Task: Add CI Gate Monitoring workflow
    - [ ] Create `.github/workflows/ci-gate-monitor.yml`
    - [ ] Trigger on workflow_run completion
    -   [ ] Check status of all workflows triggered by the same push
    -   [ ] Post comment on PR or push notification if any fail
    -   [ ] Auto-create issue for persistent CI failures
- [ ] Task: Delete obsolete CI workflows
    - [ ] Remove old ci.yml, python_ci.yml, lint.yml
- [ ] Task: Conductor - Automated Review 'Phase 5' (Protocol in workflow.md)

## Phase 6: Replace Dependabot with Renovate

- [ ] Task: Create Renovate configuration
    - [ ] Create `.github/renovate.json` with:
        - Weekly schedule (Monday)
        - Group all dev dependencies together
        - Group all production dependencies together
        - Group GitHub Actions updates together
        - Enable auto-merge for patch updates that pass CI
        - Include changelogs in PR descriptions
        - Label dependency PRs with `dependencies`
        - Pin digest for GitHub Actions
    - [ ] Test Renovate configuration with `renovate --dry-run` if possible
- [ ] Task: Delete Dependabot
    - [ ] Remove `.github/dependabot.yml`
- [ ] Task: Install Renovate GitHub App
    - [ ] Document the Renovate app installation URL for the repository
- [ ] Task: Conductor - Automated Review 'Phase 6' (Protocol in workflow.md)

## Phase 7: Add SOTA Files and Infrastructure

- [ ] Task: Create CITATION.cff
    - [ ] Add authors, title, abstract, version, DOI (if available), URL, license
    - [ ] Include preferred citation format for academic papers
    - [ ] Validate with cff-validator
- [ ] Task: Create CODEOWNERS
    - [ ] Set repository owners for code, docs, and CI
    - [ ] Require review from owners for changes to critical paths
- [ ] Task: Create SECURITY.md
    - [ ] Define security reporting process
    - [ ] List supported versions that receive security patches
    - [ ] Provide contact method for security issues
- [ ] Task: Create .editorconfig
    - [ ] Define consistent indentation, line endings, charset
    - [ ] Align with Ruff formatting rules
- [ ] Task: Add actionlint CI job
    - [ ] Add actionlint to the lint job in CI workflow
    - [ ] Validate all GitHub Actions workflows for errors
- [ ] Task: Add release-please workflow
    - [ ] Create `.github/workflows/release-please.yml`
    - [ ] Configure for Python/hatch or setuptools
    - [ ] Auto-generate changelog from conventional commits
    - [ ] Auto-create GitHub releases
    - [ ] Replace release-drafter.yml
- [ ] Task: Add Scalene profiling setup
    - [ ] Add Scalene to dev dependencies
    - [ ] Create `scripts/profile.py` for easy profiling
    - [ ] Add profiling instructions to README.md
- [ ] Task: Conductor - Automated Review 'Phase 7' (Protocol in workflow.md)

## Phase 8: Test Structure Reorganization

- [ ] Task: Reorganize test directory structure
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
