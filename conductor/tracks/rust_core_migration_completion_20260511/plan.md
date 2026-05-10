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

## Phase 2: Promote the Remaining Canonical Operations and Model Families [checkpoint: 9a77de3]

- [x] Task: Implement Rust-native execution for the remaining promoted slices
    - [x] Add native implementations or explicit non-Python promotions for each remaining canonical operation
    - [x] Add native model-family slices for the remaining registry families
    - [x] Keep the Python bridge only where the slice is intentionally non-native
- [x] Task: Add parity and regression coverage for every promoted slice
    - [x] Write native-vs-bridge parity tests
    - [x] Write model-family regression tests for the newly promoted slices
    - [x] Verify fitted-state, prediction, simulation, summary, and diagnostics behavior
- [x] Task: Prove the promoted slices are stable
    - [x] Add benchmark or profiling evidence for each promoted slice
    - [x] Confirm deterministic error mapping and payload decoding
    - [x] Confirm unsupported shapes fail with explicit capability errors
- [x] Task: Conductor - Automated Review and Checkpoint 'Promote the Remaining Canonical Operations and Model Families' (Protocol in workflow.md)

## Phase 3: Remove Undocumented Fallback and Lock the ABI Boundary [checkpoint: a343566]

- [x] Task: Narrow bridge fallback to only intentionally non-native surfaces
    - [x] Remove bridge fallback from promoted slices
    - [x] Keep fallback only for explicitly documented exceptions
    - [x] Verify native-first routing cannot silently drift to Python
- [x] Task: Update ABI and capability metadata
    - [x] Refresh ABI policy docs and schema-version notes
    - [x] Tighten capability discovery to reflect final ownership
    - [x] Add tests that reject exposure of Python internals or private native structs
- [x] Task: Validate cross-language contract stability
    - [x] Confirm Python, Rust, R, Julia, Go, TypeScript, and C# still bind to the same kernel contract
    - [x] Add contract tests for changed request and response shapes
    - [x] Confirm bridge-only behavior is explicit in the remaining fallback paths
- [x] Task: Conductor - Automated Review and Checkpoint 'Remove Undocumented Fallback and Lock the ABI Boundary' (Protocol in workflow.md)

## Phase 4: Align Bindings, Packaging, and Publication Surfaces

- [x] Task: Update binding documentation and manifests
    - [x] Refresh the Rust binding README and package docs
    - [x] Refresh the Python-facing binding docs if ownership wording changed
    - [x] Refresh R, Julia, Go, TypeScript, and C# docs or manifests that describe runtime ownership
- [x] Task: Align publication and smoke tests
    - [x] Ensure package publish gates still exercise the promoted slices
    - [x] Verify binding smoke tests cover native-first execution
    - [x] Update release notes or CRAN/NuGet/npm metadata if needed
- [x] Task: Keep community-facing evidence aligned
    - [x] Update reviewer-facing evidence pages that describe the core architecture
    - [x] Update any release or community docs that still imply unresolved Rust ownership gaps
    - [x] Keep package-manager metadata and docs synchronized with the new core claim
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
