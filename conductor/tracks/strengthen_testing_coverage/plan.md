# Implementation Plan: Strengthen Testing Infrastructure and Core Model Coverage

## Phase 1: Coverage Analysis and Gap Identification

- [x] Task: Run baseline coverage analysis for all core modules `a6a4188`
    - [x] Execute coverage analysis `a6a4188`
    - [x] Document coverage gaps (uncovered lines, branches) for each module `a6a4188`
    - [x] Identify critical missing test scenarios (edge cases, error paths) `a6a4188`
- [x] Task: Review existing test files for quality and completeness `708cd87`
    - [x] Analyze test patterns and conventions used across the test suite `708cd87`
    - [x] Identify tests that need updating vs. tests that need creation `708cd87`
    - [x] Document the testing conventions (naming, fixtures, parametrization) `708cd87`
- [x] Task: Conductor - Automated Review 'Phase 1' (Protocol in workflow.md) `708cd87`

## Phase 2: Diffusion Model Test Strengthening

- [x] Task: Write comprehensive tests for Bass model `708cd87`
    - [x] Test basic fit/predict/simulate workflows `708cd87`
    - [x] Test covariate-driven parameter fitting `708cd87`
    - [x] Test time-varying parameter functionality `708cd87`
    - [x] Test mixture model fitting `708cd87`
    - [x] Test edge cases: zero adoption data, single data point, perfect fit scenarios `708cd87`
    - [x] Test error handling: invalid inputs, convergence failures `708cd87`
- [x] Task: Write comprehensive tests for Gompertz model `708cd87`
    - [x] Test basic fit/predict/simulate workflows `708cd87`
    - [x] Test covariate and time-varying parameters `708cd87`
    - [x] Test mixture model fitting `708cd87`
    - [x] Test edge cases and error handling `708cd87`
- [x] Task: Write comprehensive tests for Logistic model `708cd87`
    - [x] Test basic fit/predict/simulate workflows `708cd87`
    - [x] Test covariate and time-varying parameters `708cd87`
    - [x] Test mixture model fitting `708cd87`
    - [x] Test edge cases and error handling `708cd87`
- [x] Task: Conductor - Automated Review 'Phase 2' (Protocol in workflow.md) `708cd87`

## Phase 3: Substitution and Competition Model Test Strengthening

- [ ] Task: Write comprehensive tests for Fisher-Pry substitution model
    - [ ] Test basic substitution curve generation
    - [ ] Test multi-generational substitution
    - [ ] Test edge cases: no substitution, complete substitution, parameter boundaries
    - [ ] Test error handling
- [ ] Task: Write comprehensive tests for Norton-Bass substitution model
    - [ ] Test basic multi-generation fitting
    - [ ] Test prediction and simulation
    - [ ] Test edge cases and error handling
- [ ] Task: Write comprehensive tests for competition models
    - [ ] Test Lotka-Volterra competition dynamics
    - [ ] Test Market Share Attraction model
    - [ ] Test Replicator Dynamics model
    - [ ] Test edge cases: single competitor, zero growth, equilibrium states
    - [ ] Test error handling
- [ ] Task: Conductor - Automated Review 'Phase 3' (Protocol in workflow.md)

## Phase 4: Remaining Core Model Test Strengthening

- [ ] Task: Write comprehensive tests for Hype Cycle models
    - [ ] Test composite hype cycle generation
    - [ ] Test DDE-based hype models
    - [ ] Test public sentiment impact simulation
    - [ ] Test edge cases and error handling
- [ ] Task: Write comprehensive tests for Fail (failure analysis) models
    - [ ] Test failure detection and analysis
    - [ ] Test adoption failure scenarios
    - [ ] Test edge cases and error handling
- [ ] Task: Write comprehensive tests for Adopt (adopter classification)
    - [ ] Test adopter type classification logic
    - [ ] Test timing-based classification
    - [ ] Test edge cases and error handling
- [ ] Task: Write comprehensive tests for Fitters module
    - [ ] Test curve fitting utilities
    - [ ] Test optimization routines
    - [ ] Test batched fitting
    - [ ] Test error handling and convergence failures
- [ ] Task: Conductor - Automated Review 'Phase 4' (Protocol in workflow.md)

## Phase 5: Property-Based Testing Infrastructure

- [ ] Task: Set up Hypothesis testing infrastructure
    - [ ] Add Hypothesis strategies for common data types (adoption curves, parameter sets, time series)
    - [ ] Create shared Hypothesis fixtures in conftest.py
- [ ] Task: Write property-based tests for diffusion models
    - [ ] Test S-curve properties: monotonicity of cumulative adoption, non-negative rates
    - [ ] Test parameter bounds: p, q in [0,1], m > 0
    - [ ] Test asymptotic behavior: adoption approaches m as t → ∞
    - [ ] Test scale invariance properties
- [ ] Task: Write property-based tests for substitution models
    - [ ] Test substitution fractions sum to 1
    - [ ] Test monotonic replacement properties
- [ ] Task: Write property-based tests for competition models
    - [ ] Test competitive exclusion principles
    - [ ] Test equilibrium stability properties
- [ ] Task: Conductor - Automated Review 'Phase 5' (Protocol in workflow.md)

## Phase 6: Mutation Testing and Integration Tests

- [ ] Task: Configure and run mutation testing with mutmut
    - [ ] Set up mutmut configuration for all core modules
    - [ ] Run initial mutation testing pass
    - [ ] Identify surviving mutants
    - [ ] Write additional tests to kill surviving mutants
    - [ ] Target: >70% mutation score
- [ ] Task: Write cross-module integration tests
    - [ ] Test end-to-end workflow: data → fit → predict → plot
    - [ ] Test diffusion → substitution handoff scenarios
    - [ ] Test competition → hype cycle combined scenarios
    - [ ] Test mixture model → prediction pipeline
- [ ] Task: Run final coverage verification
    - [ ] Execute full test suite with coverage
    - [ ] Verify all modules meet >90% coverage threshold
    - [ ] Document any remaining gaps with justification
- [ ] Task: Conductor - Automated Review 'Phase 6' (Protocol in workflow.md)

## Phase 7: Quality Gate and Documentation

- [ ] Task: Run complete quality gate verification
    - [ ] All tests pass (`uv run pytest`)
    - [ ] Coverage >80% overall, >90% for core modules
    - [ ] No linting errors (`uv run ruff check .`)
    - [ ] Type checking passes (`uv run ty check src/`)
    - [ ] Security scanning passes (`uv run bandit -r src/innovate`)
- [ ] Task: Update test documentation
    - [ ] Document new test patterns and conventions
    - [ ] Document Hypothesis strategies and usage
    - [ ] Document mutation testing setup and interpretation
- [ ] Task: Final review and cleanup
    - [ ] Review all changes for code quality
    - [ ] Ensure no debug code or temporary files remain
    - [ ] Verify all git notes and plan updates are complete
- [ ] Task: Conductor - Automated Review 'Phase 7' (Protocol in workflow.md)
