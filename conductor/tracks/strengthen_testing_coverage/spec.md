# Track Specification: Strengthen Testing Infrastructure and Core Model Coverage

## Overview
This track focuses on strengthening the testing infrastructure and ensuring comprehensive test coverage across all core modules of the innovate library.

## Objectives
1. Ensure all core diffusion models (Bass, Gompertz, Logistic) have >90% test coverage
2. Ensure all substitution models (Fisher-Pry, Norton-Bass) have >90% test coverage
3. Ensure all competition models have >90% test coverage
4. Add property-based tests using Hypothesis for mathematical invariants
5. Set up and run mutation testing with mutmut
6. Improve integration test coverage for cross-module workflows
7. Ensure all public APIs have comprehensive edge case testing

## Scope
### In Scope
- `src/innovate/diffuse/` — All diffusion model implementations
- `src/innovate/substitute/` — All substitution model implementations
- `src/innovate/compete/` — All competition model implementations
- `src/innovate/hype/` — Hype cycle models
- `src/innovate/fail/` — Failure analysis models
- `src/innovate/adopt/` — Adopter classification
- `src/innovate/fitters/` — Parameter fitting utilities
- `src/innovate/base/` — Base classes and abstractions
- Test infrastructure (pytest config, fixtures, conftest.py)
- Property-based testing infrastructure (Hypothesis)
- Mutation testing setup (mutmut)

### Out of Scope
- New feature development
- ABM module enhancements
- JAX backend optimization
- Documentation updates (beyond test-related docs)
- CI/CD pipeline changes

## Acceptance Criteria
- All modules listed in scope have >90% code coverage
- Property-based tests cover mathematical invariants for all core models
- Mutation testing score (mutmut) is >70%
- All existing tests continue to pass
- No regression in test execution time
- Integration tests cover end-to-end workflows for diffusion → fitting → prediction → plotting

## Technical Approach
1. **Coverage Analysis**: Run current coverage reports to identify gaps
2. **Test Augmentation**: Write targeted tests for uncovered branches and edge cases
3. **Property-Based Testing**: Define and test mathematical invariants (e.g., S-curve monotonicity, parameter bounds, asymptotic behavior)
4. **Mutation Testing**: Configure and run mutmut, strengthen tests to kill mutants
5. **Integration Tests**: Add cross-module integration tests
6. **Verification**: Final coverage report and quality gate verification
