# Product Guidelines

## Prose Style
- **Clarity First**: Write documentation that is accessible to both domain experts and newcomers. Avoid unnecessary jargon; when domain-specific terms are required, define them on first use.
- **Concise and Direct**: Prefer short, declarative sentences. Eliminate filler words. Every sentence should carry information.
- **Active Voice**: Use active voice in documentation, docstrings, and code comments (e.g., "The model fits the data" not "The data is fitted by the model").
- **Value Prose Linting**: Governance prose is checked with Vale, using the `Repo/ValueProse` style to catch hedging and filler in repo policy docs.

## Code Documentation
- **Docstrings**: Every public class and function must have a NumPy-style docstring containing:
  - A one-line summary
  - Extended description (if needed)
  - `Parameters` section with types and descriptions
  - `Returns` section with type and description
  - `Examples` section with runnable code snippets
- **Type Hints**: All public APIs must be fully type-hinted. Use `typing_extensions` for forward compatibility. Type checking is enforced by `ty` (primary) and `mypy` (secondary).
- **Inline Comments**: Comment *why*, not *what*. Only add inline comments for non-obvious logic, mathematical derivations, or performance-critical sections.

## API Design Principles
- **Consistency**: Model APIs should follow a uniform pattern: `fit()`, `predict()`, `simulate()`, `plot()`.
- **Sensible Defaults**: Provide well-researched default parameters so that users can get meaningful results with minimal configuration.
- **Progressive Disclosure**: Simple use cases should require minimal code; advanced features (covariates, time-varying params, mixtures) should be accessible through optional keyword arguments.
- **Backward Compatibility**: Public API changes must be additive. Breaking changes require a major version bump and a migration guide.

## Visualization Guidelines
- **Publication-Ready**: All plots should be clean, well-labeled, and suitable for inclusion in academic papers without post-processing.
- **Accessible Defaults**: Use colorblind-friendly palettes. Ensure sufficient contrast and legible font sizes.
- **Customizable**: Expose matplotlib `Axes` objects so users can further customize any plot.

## Testing Principles
- **Test-Driven**: New features should be accompanied by tests before or during implementation.
- **Three-Tier Structure**:
  - **Unit Tests** (`tests/unit/`): Individual function/class isolation. Run with `pytest -m unit`.
  - **Integration Tests** (`tests/integration/`): Cross-module interactions. Run with `pytest -m integration`.
  - **End-to-End Tests** (`tests/e2e/`): Complete user workflows. Run with `pytest -m e2e`.
- **Coverage Threshold**: Maintain >80% code coverage overall. Critical modules (diffuse, compete, substitute, fitters) should target >90%.
- **Property-Based Testing**: Use `hypothesis` for testing mathematical invariants and edge cases in model behavior.
- **Mutation Testing**: Run `mutmut` weekly via CI to assess test quality alongside coverage. Target >70% mutation score.

## Performance
- **Vectorization First**: Prefer NumPy vectorized operations over Python loops.
- **Backend Abstraction**: Computational backends (NumPy, JAX) must be swappable without changing user-facing API.
- **Benchmarking**: Performance-sensitive changes should include benchmark tests via `pytest-benchmark`.
- **Profiling**: Use **Scalene** for CPU, memory, and GPU profiling with per-line attribution. Profile critical paths before and after optimization.

## Error Handling
- **Informative Errors**: Raise specific exception types with clear messages that tell the user what went wrong and how to fix it.
- **Fail Fast**: Validate inputs early and raise errors before any expensive computation begins.
- **Warnings vs. Errors**: Use `warnings.warn()` for deprecated APIs and recoverable conditions; raise exceptions for invalid inputs or configurations.

## Dependency Management
- **uv**: All dependency management uses `uv`. Commands: `uv sync`, `uv add <pkg>`, `uv lock`.
- **Lockfile**: `uv.lock` is committed and serves as the source of truth for reproducible builds.
- **Automated Updates**: Renovate handles dependency updates with intelligent grouping. Patch updates auto-merge if CI passes.
- **Version Pinning**: Production dependencies use compatible release ranges (`>=x.y, <x+1`). Dev dependencies use latest compatible versions.

## Linting & Formatting
- **Ruff is the Single Tool**: All linting and formatting is handled by Ruff.
  - `uv run ruff check .` — Lint (replaces flake8, Pylint, vulture, unimport)
  - `uv run ruff format .` — Format (replaces Black)
  - `uv run ruff check . --select I` — Import sorting (replaces isort)
- **Pre-commit Hooks**: All commits are validated by pre-commit hooks running Ruff, mypy, codespell, and safety checks.
- **No Legacy Tools**: Black, isort, flake8, Pylint, vulture, and unimport are not used.

## Type Safety
- **Primary**: `ty` is the primary type checker (fastest, by Astral).
- **Secondary**: `mypy` runs in strict mode as a secondary check in CI.
- **Enforcement**: All CI checks must pass type checking before code can be merged.

## Versioning & Releases
- **Semantic Versioning**: Follow `MAJOR.MINOR.PATCH`. Breaking changes increment MAJOR.
- **Conventional Commits**: All commits follow the conventional commit format (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`, `perf:`, `refactor:`).
- **Automated Releases**: `release-please` automatically generates changelogs, creates releases, and bumps versions based on conventional commits.
- **CHANGELOG**: Auto-generated by release-please from conventional commit history.

## CI/CD
- **Single Pipeline**: All CI checks run in a consolidated GitHub Actions workflow with parallel jobs.
- **CI Gate Monitoring**: After every push, CI results are automatically monitored. Failures are addressed iteratively until all checks pass.
- **Matrix Testing**: Tests run on Python 3.10, 3.11, 3.12, 3.13.
- **Coverage**: Codecov integration with PR comments on coverage changes.
- **Scheduled Jobs**: Mutation testing runs weekly. Dependency updates via Renovate run weekly.

## Code Review Process
- **Automated Review**: The `conductor:review` skill performs automated code review at the end of every track phase. It checks plan compliance, style compliance, correctness, testing, and security — then automatically applies fixes.
- **CI as Gate**: All code must pass the full CI pipeline before merging. No manual review is required unless the automated review fails repeatedly.

## Citation
- **CITATION.cff**: The repository includes a `CITATION.cff` file for academic citation. All publications using this library should cite it.
- **Preferred Citation**: Include the library name, version, authors, and DOI (if available) in academic papers.
