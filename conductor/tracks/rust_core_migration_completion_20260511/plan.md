# Implementation Plan: Rust Core Migration Completion and Polyglot Claim Closure

## Phase 1: Audit the Remaining Rust-Core Gap [checkpoint: 6c83eb8]

- [x] Task: Build a complete ownership inventory
    - [x] Enumerate every canonical operation and stable payload shape
    - [x] Enumerate every model family still routed through bridge fallback
    - [x] Record which slices are Rust-native, bridge-backed, or Python-reference
- [x] Task: Write failing tests for the current gap
    - [x] Add inventory completeness checks
    - [x] Add tests that assert the roadmap still exposes unresolved bridge-backed slices
    - [x] Add tests that fail when promoted slices can still reach undocumented fallback
- [x] Task: Capture the migration baseline in docs and fixtures
    - [x] Refresh the machine-readable Rust-core inventory fixture
    - [x] Update any stale roadmap assertions that misstate current ownership
    - [x] Record the exact model families and operations that remain to be migrated
- [x] Task: Conductor - Automated Review and Checkpoint 'Audit the Remaining Rust-Core Gap' (Protocol in workflow.md)

## Phase 2: Promote the Remaining Canonical Operations and Model Families

- [ ] Task: Implement Rust-native execution for the remaining promoted slices
    - [ ] Add native implementations or explicit non-Python promotions for each remaining canonical operation
    - [ ] Add native model-family slices for the remaining registry families
    - [ ] Keep the Python bridge only where the slice is intentionally non-native
- [ ] Task: Add parity and regression coverage for every promoted slice
    - [ ] Write native-vs-bridge parity tests
    - [ ] Write model-family regression tests for the newly promoted slices
    - [ ] Verify fitted-state, prediction, simulation, summary, and diagnostics behavior
- [ ] Task: Prove the promoted slices are stable
    - [ ] Add benchmark or profiling evidence for each promoted slice
    - [ ] Confirm deterministic error mapping and payload decoding
    - [ ] Confirm unsupported shapes fail with explicit capability errors
- [ ] Task: Conductor - Automated Review and Checkpoint 'Promote the Remaining Canonical Operations and Model Families' (Protocol in workflow.md)

## Phase 3: Remove Undocumented Fallback and Lock the ABI Boundary

- [ ] Task: Narrow bridge fallback to only intentionally non-native surfaces
    - [ ] Remove bridge fallback from promoted slices
    - [ ] Keep fallback only for explicitly documented exceptions
    - [ ] Verify native-first routing cannot silently drift to Python
- [ ] Task: Update ABI and capability metadata
    - [ ] Refresh ABI policy docs and schema-version notes
    - [ ] Tighten capability discovery to reflect final ownership
    - [ ] Add tests that reject exposure of Python internals or private native structs
- [ ] Task: Validate cross-language contract stability
    - [ ] Confirm Python, Rust, R, Julia, Go, TypeScript, and C# still bind to the same kernel contract
    - [ ] Add contract tests for changed request and response shapes
    - [ ] Confirm bridge-only behavior is explicit in the remaining fallback paths
- [ ] Task: Conductor - Automated Review and Checkpoint 'Remove Undocumented Fallback and Lock the ABI Boundary' (Protocol in workflow.md)

## Phase 4: Align Bindings, Packaging, and Publication Surfaces

- [ ] Task: Update binding documentation and manifests
    - [ ] Refresh the Rust binding README and package docs
    - [ ] Refresh the Python-facing binding docs if ownership wording changed
    - [ ] Refresh R, Julia, Go, TypeScript, and C# docs or manifests that describe runtime ownership
- [ ] Task: Align publication and smoke tests
    - [ ] Ensure package publish gates still exercise the promoted slices
    - [ ] Verify binding smoke tests cover native-first execution
    - [ ] Update release notes or CRAN/NuGet/npm metadata if needed
- [ ] Task: Keep community-facing evidence aligned
    - [ ] Update reviewer-facing evidence pages that describe the core architecture
    - [ ] Update any release or community docs that still imply unresolved Rust ownership gaps
    - [ ] Keep package-manager metadata and docs synchronized with the new core claim
- [ ] Task: Conductor - Automated Review and Checkpoint 'Align Bindings, Packaging, and Publication Surfaces' (Protocol in workflow.md)

## Phase 5: Update Architecture, Roadmap, and Readiness Claims

- [ ] Task: Rewrite the Rust core roadmap to the new ownership state
    - [ ] Remove stale bridge-gap language for promoted slices
    - [ ] Mark any permanently non-native slices explicitly
    - [ ] Keep the operation inventory fixture as the machine-readable source of truth
- [ ] Task: Update polyglot architecture and readiness docs
    - [ ] Refresh the polyglot repository architecture page
    - [ ] Refresh the scientific and HPC readiness roadmap
    - [ ] Refresh any submission-readiness or governance docs that reference Rust-core ownership
- [ ] Task: Update the main README and canonical docs
    - [ ] Align the short-form README with the final migration claim
    - [ ] Align the docs landing pages and binding overview pages
    - [ ] Ensure no page overstates the current ownership state
- [ ] Task: Conductor - Automated Review and Checkpoint 'Update Architecture, Roadmap, and Readiness Claims' (Protocol in workflow.md)

## Phase 6: Validate the Full Migration and Archive the Track

- [ ] Task: Run the complete validation matrix
    - [ ] Run focused Rust-core parity and regression tests
    - [ ] Run binding tests across supported languages
    - [ ] Run docs build, lint, and type checks
    - [ ] Run benchmark or profiling validations required by the promotion gates
- [ ] Task: Run final conductor review
    - [ ] Review the full diff against the spec, plan, workflow, and tests
    - [ ] Apply any high-confidence fixes surfaced by the review
    - [ ] Re-run validation until the track is stable
- [ ] Task: Archive the completed migration track
    - [ ] Move the track folder to the archive location
    - [ ] Update the tracks registry entry to completed
    - [ ] Preserve links to the final evidence bundle
- [ ] Task: Conductor - Automated Review and Checkpoint 'Validate the Full Migration and Archive the Track' (Protocol in workflow.md)
