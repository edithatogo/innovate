# Product Guidelines

## Prose Style
- **Clarity First**: Write documentation that is accessible to both domain experts and newcomers. Avoid unnecessary jargon; when domain-specific terms are required, define them on first use.
- **Concise and Direct**: Prefer short, declarative sentences. Eliminate filler words. Every sentence should carry information.
- **Active Voice**: Use active voice in documentation, docstrings, and code comments (e.g., "The model fits the data" not "The data is fitted by the model").

## Code Documentation
- **Docstrings**: Every public class and function must have a NumPy-style docstring containing:
  - A one-line summary
  - Extended description (if needed)
  - `Parameters` section with types and descriptions
  - `Returns` section with type and description
  - `Examples` section with runnable code snippets
- **Type Hints**: All public APIs must be fully type-hinted. Use `typing_extensions` for forward compatibility.
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
- **Coverage Threshold**: Maintain >80% code coverage. Critical modules (diffuse, compete, substitute, fitters) should target >90%.
- **Property-Based Testing**: Use `hypothesis` for testing mathematical invariants and edge cases in model behavior.
- **Mutation Testing**: Periodically run `mutmut` to assess test quality, not just coverage quantity.

## Performance
- **Vectorization First**: Prefer NumPy vectorized operations over Python loops.
- **Backend Abstraction**: Computational backends (NumPy, JAX) must be swappable without changing user-facing API.
- **Benchmarking**: Performance-sensitive changes should include benchmark tests via `pytest-benchmark`.

## Error Handling
- **Informative Errors**: Raise specific exception types with clear messages that tell the user what went wrong and how to fix it.
- **Fail Fast**: Validate inputs early and raise errors before any expensive computation begins.
- **Warnings vs. Errors**: Use `warnings.warn()` for deprecated APIs and recoverable conditions; raise exceptions for invalid inputs or configurations.

## Versioning & Releases
- **Semantic Versioning**: Follow `MAJOR.MINOR.PATCH`. Breaking changes increment MAJOR.
- **CHANGELOG**: Maintain a human-readable `CHANGELOG.md` with entries grouped by release.
- **Release Cadence**: Coordinate releases with meaningful feature milestones or critical bug fixes.
