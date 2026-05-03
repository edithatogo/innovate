# Implementation Plan: C# Package Publication

## Phase 1: Contract and Package Readiness

- [ ] Task: Validate C# binding contract
    - [ ] Verify C# schema compatibility fixtures against the functional kernel
    - [ ] Confirm C# remains a thin binding with no duplicated model logic
    - [ ] Identify missing .NET 10/.NET 11 project or packaging settings
- [ ] Task: Define NuGet package metadata and release gates
    - [ ] Specify package ID, versioning, license, README, source-link, and repository metadata
    - [ ] Define signing, provenance, artifact retention, and rollback expectations
    - [ ] Document when publishing is allowed versus dry-run only
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Contract and Package Readiness' (Protocol in workflow.md)

## Phase 2: CI and Packaging

- [ ] Task: Add C# publication readiness tests
    - [ ] Write failing checks for package metadata and .NET 10/.NET 11 targeting
    - [ ] Write failing checks for pack artifacts and schema fixture compatibility
    - [ ] Add checks that publication remains gated outside release contexts
- [ ] Task: Implement NuGet packaging workflow
    - [ ] Configure restore, build, test, pack, and dry-run publish jobs
    - [ ] Add package metadata and release documentation
    - [ ] Ensure secrets are only required for release publication
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: CI and Packaging' (Protocol in workflow.md)

## Phase 3: Documentation and Validation

- [ ] Task: Document C# publication workflow
    - [ ] Explain local package validation and release commands
    - [ ] Document NuGet publication gates and rollback steps
    - [ ] Cross-link C# binding contract and ecosystem versioning docs
- [ ] Task: Run validation gates
    - [ ] Run C# restore, build, test, and pack checks
    - [ ] Run multi-language schema compatibility checks
    - [ ] Run relevant lint, type, and documentation checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Validation' (Protocol in workflow.md)
