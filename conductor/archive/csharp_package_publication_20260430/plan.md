# Implementation Plan: C# Package Publication

## Phase 1: Contract and Package Readiness

- [x] Task: Validate C# binding contract
    - [x] Verify C# schema compatibility fixtures against the functional kernel
    - [x] Confirm C# remains a thin binding with no duplicated model logic
    - [x] Identify missing .NET 10/.NET 11 project or packaging settings
- [x] Task: Define NuGet package metadata and release gates
    - [x] Specify package ID, versioning, license, README, source-link, and repository metadata
    - [x] Define signing, provenance, artifact retention, and rollback expectations
    - [x] Document when publishing is allowed versus dry-run only
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Contract and Package Readiness' (Protocol in workflow.md)

## Phase 2: CI and Packaging

- [x] Task: Add C# publication readiness tests
    - [x] Write failing checks for package metadata and .NET 10/.NET 11 targeting
    - [x] Write failing checks for pack artifacts and schema fixture compatibility
    - [x] Add checks that publication remains gated outside release contexts
- [x] Task: Implement NuGet packaging workflow
    - [x] Configure restore, build, test, pack, and dry-run publish jobs
    - [x] Add package metadata and release documentation
    - [x] Ensure secrets are only required for release publication
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: CI and Packaging' (Protocol in workflow.md)

## Phase 3: Documentation and Validation

- [x] Task: Document C# publication workflow
    - [x] Explain local package validation and release commands
    - [x] Document NuGet publication gates and rollback steps
    - [x] Cross-link C# binding contract and ecosystem versioning docs
- [x] Task: Run validation gates
    - [x] Run C# restore, build, test, and pack checks
    - [x] Run multi-language schema compatibility checks
    - [x] Run relevant lint, type, and documentation checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Validation' (Protocol in workflow.md)
