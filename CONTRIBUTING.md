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

#### Pull Request Process

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Add your changes and tests
3. Run tests: `pytest`
4. Add documentation as needed
5. Run pre-commit: `pre-commit run --all-files`
6. Create the pull request

### Style Guidelines

#### Code Style
- Follow PEP 8 style guide
- Use Black for code formatting
- Use type hints for all public methods/functions
- Write docstrings using numpy documentation style

#### Test Guidelines
- Add unit tests for new functionality
- Aim for high test coverage
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
uv run pytest
```

Run tests with coverage:
```bash
uv run pytest --cov=innovate --cov-report=html
```

Run unit tests only (fast feedback):
```bash
uv run pytest -m unit
```

### Linting and Formatting
The project uses Ruff for all linting and formatting, enforced via pre-commit hooks:
```bash
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run ruff check . --fix  # Auto-fix
```

Type checking:
```bash
uv run ty check src/       # Primary type checker
uv run mypy src/           # Secondary (strict mode)
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