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
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install in development mode: `pip install -e .`
6. Install pre-commit hooks: `pre-commit install`

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
pytest
```

Run tests with coverage:
```bash
pytest --cov=innovate --cov-report=html
```

### Linting and Formatting
The project uses pre-commit hooks to enforce code quality. Ensure you've installed them:
```bash
pre-commit install
```

Manual formatting and linting:
```bash
black .
isort .
flake8
mypy src/
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