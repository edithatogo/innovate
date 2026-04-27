# Contributing to innovate

Thank you for your interest in contributing to the `innovate` library! This document outlines the process for contributing.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Use the issue tracker to report bugs
- Follow the provided issue template
- Include a clear description, steps to reproduce, and expected vs actual behavior

### Suggesting Features

- Use the issue tracker to suggest features
- Explain the use case and potential implementation approach
- Consider the library's scope and design philosophy

### Code Contributions

#### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/innovate.git`
3. Install dependencies with uv: `uv sync --all-extras`
4. Install pre-commit hooks: `uv run pre-commit install`

If you are using Codex with the repo-managed Conductor workflow, sync the project-owned skills into your Codex home:

```bash
uv run python scripts/sync_codex_skills.py
```

#### Pull Request Process

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Add your changes and tests
3. Run the verified local quality gates listed below
4. Add documentation as needed
5. Run pre-commit: `uv run pre-commit run --all-files`
6. Create the pull request

### Style Guidelines

#### Code Style
- Follow PEP 8 style guide
- Use Ruff for linting and formatting
- Use type hints for all public methods/functions
- Write docstrings using numpy documentation style

#### Test Guidelines
- Add unit tests for new functionality
- Treat coverage as a diagnostic signal unless a task explicitly raises a threshold
- Follow the existing test patterns
- Test edge cases and error conditions

#### Documentation
- Update docstrings to reflect changes
- Add examples where appropriate
- Keep README updated
- Add comprehensive API documentation

## Development Workflow

### Testing
Run all tests with:
```bash
CI=true uv run python -m pytest
```

Run the required CI unit gate locally:
```bash
CI=true uv run python -m pytest tests/unit/ \
  --ignore=tests/unit/test_bayesian_fitter_robust.py \
  --ignore=tests/unit/test_blackjax_fitter.py \
  --ignore=tests/unit/test_dynamics_competition_direct.py \
  --ignore=tests/unit/test_dynamics_contagion_direct.py \
  --ignore=tests/unit/test_categorization.py \
  --ignore=tests/unit/test_path_dependence.py \
  --ignore=tests/unit/test_preprocess.py \
  --ignore=tests/unit/test_plots_diagnostics.py \
  --ignore=tests/unit/test_plots_network.py \
  --ignore=tests/unit/test_advanced_functionality_comprehensive.py \
  --ignore=tests/unit/test_curve_fitter.py \
  --ignore=tests/unit/test_fitters.py \
  --ignore=tests/unit/test_batched_fitter.py \
  --cov=innovate \
  --cov-report=term-missing \
  --cov-fail-under=0 \
  --tb=short \
  -q
```

Run tests with a coverage report for inspection:
```bash
CI=true uv run python -m pytest --cov=innovate --cov-report=html
```

### Linting and Formatting
The project uses Ruff for all linting and formatting, enforced via pre-commit hooks:
```bash
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run ruff check . --fix  # Auto-fix
```

### Codex Skill Sync

The autonomous Conductor workflow for this repository is defined by the vendored skill copies in `.codex/skills/`. Refresh your local Codex installation from the repo whenever those files change:

```bash
uv run python scripts/sync_codex_skills.py
```

To inspect what would be installed without modifying your Codex home:

```bash
uv run python scripts/sync_codex_skills.py --dry-run
```

The repo-managed bundle includes the project setup, status, revert, new-track, implementation, and review skills, so the full Conductor surface is reproducible from source control.

Type checking:
```bash
uv run ty check \
  src/innovate/__init__.py \
  src/innovate/backend.py \
  src/innovate/backends/__init__.py \
  src/innovate/capabilities.py \
  src/innovate/diffuse/__init__.py \
  src/innovate/substitute/__init__.py \
  src/innovate/ecosystem/__init__.py

uv run mypy \
  src/innovate/__init__.py \
  src/innovate/backend.py \
  src/innovate/backends/__init__.py \
  src/innovate/capabilities.py \
  src/innovate/diffuse/__init__.py \
  src/innovate/substitute/__init__.py \
  src/innovate/ecosystem/__init__.py
```

Package and docs smoke checks:
```bash
uv build
uv run python -m sphinx -b html docs/source /tmp/innovate-docs-build
```

## Project Structure

- `src/innovate/`: Main source code
  - `diffuse/`: Single innovation diffusion models
  - `compete/`: Competition models
  - `substitute/`: Substitution models
  - `hype/`: Hype cycle models
  - `base/`: Base classes
  - `fitters/`: Parameter fitting algorithms
  - `dynamics/`: Dynamics models (contagion, competition)
  - `utils/`: Utility functions
- `tests/`: Test files
- `examples/`: Example notebooks and scripts
- `docs/`: Documentation

## Getting Help

- Open an issue for technical questions
- Check existing issues and pull requests for similar problems
- Contact maintainers if you need clarification

## Acknowledgements

Thank you to all our contributors! Your efforts help improve the library for everyone.
