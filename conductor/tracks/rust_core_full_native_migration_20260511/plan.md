# Implementation Plan: Rust Core Full Native Migration and Ownership Closure

## Phase 1: Map the Remaining Rust-Core Ownership Gaps [checkpoint: 15e6004]

- [x] Task: Inventory every remaining canonical operation and model family
    - [x] Record the current owner for each inventory slice
    - [x] Classify bridge-backed, Python-reference, and Rust-native slices
    - [x] Identify the stable payload shapes that still need explicit ownership
- [x] Task: Maintain the machine-readable slice ledger
    - [x] Keep the ledger synchronized with the rendered roadmap and inventory
    - [x] Record every canonical operation, model family, and stable payload shape
    - [x] Record terminal ownership as `rust_native`, `explicitly_promoted_elsewhere`, or `python_reference`
- [x] Task: Write regression tests for the ownership inventory
    - [x] Require every canonical operation to have a terminal ownership state
    - [x] Require every Python registry model family to be classified
    - [x] Require the inventory to distinguish explicit promotion from implicit fallback
    - [x] Require claim-language reconciliation once the ledger is closed
- [x] Task: Conductor - Automated Review and Checkpoint 'Map the Remaining Rust-Core Ownership Gaps' (Protocol in workflow.md)

## Phase 2: Promote or Explicitly Reassign Remaining Slices [checkpoint: 373a223]

- [x] Task: Implement native Rust execution for promotable slices
    - [x] Promote remaining core operations that can be made native
    - [x] Promote remaining diffusion-family slices that can be expressed natively
    - [x] Promote remaining competition-family slices that can be expressed natively
    - [x] Promote remaining payload and schema reconciliation slices that can be expressed natively
    - [x] Remove undocumented bridge fallback for promoted slices
- [x] Task: Record explicit non-Rust ownership where native promotion is not appropriate
    - [x] Preserve Python-reference ownership only where the contract requires it
    - [x] Document any families that remain explicitly promoted elsewhere
    - [x] Keep bridge fallback limited to documented non-native paths
- [x] Task: Conductor - Automated Review and Checkpoint 'Promote or Explicitly Reassign Remaining Slices' (Protocol in workflow.md)

## Phase 3: Prove Parity, Profiling, and Binding Stability [checkpoint: 373a223]

- [x] Task: Add and update parity and regression tests
    - [x] Test promoted slices against Python reference semantics
    - [x] Test error mapping and unsupported-payload behavior
    - [x] Test fallback routing for explicitly non-native families
- [x] Task: Build a binding smoke matrix for each promoted family
    - [x] Cover Rust and Python for every promoted slice
    - [x] Cover applicable downstream bindings: R, Julia, TypeScript, Go, and C#
    - [x] Record the matrix in the evidence bundle and ledger
- [x] Task: Capture performance and profiling evidence
    - [x] Capture benchmark output for promoted slices
    - [x] Capture CPU profiling and memory profiling where relevant
    - [x] Capture any XLA or accelerator evidence that applies
    - [x] Attach family-level profiling evidence to the ledger or evidence bundle
- [x] Task: Validate all binding surfaces
    - [x] Run Rust binding smoke coverage
    - [x] Run Python, R, Julia, TypeScript, Go, and C# smoke checks as applicable
    - [x] Preserve the resulting evidence bundle in the repo
- [x] Task: Conductor - Automated Review and Checkpoint 'Prove Parity, Profiling, and Binding Stability' (Protocol in workflow.md)

## Phase 4: Reconcile Docs, Inventory, and Claim Language [checkpoint: 373a223]

- [x] Task: Update the Rust roadmap and related docs
    - [x] Reflect the final ownership boundary in the roadmap text
    - [x] Update architecture or binding docs if claim language changes
    - [x] Remove any roadmap wording that overstates bridge ownership
- [x] Task: Reconcile claim language across docs and inventory
    - [x] Remove "partial" or equivalent wording once the ledger says full ownership is closed
    - [x] Keep the archived closure track available as evidence, not as the active source of truth
- [x] Task: Refresh the machine-readable migration inventory
    - [x] Set terminal ownership for every slice
    - [x] Record promotion blockers and rationale where needed
    - [x] Keep the docs and inventory synchronized
- [x] Task: Add regression tests for the reconciled state
    - [x] Test that the roadmap matches the inventory
    - [x] Test that no page claims full Rust ownership without evidence
    - [x] Test that the inventory and docs remain aligned
- [x] Task: Conductor - Automated Review and Checkpoint 'Reconcile Docs, Inventory, and Claim Language' (Protocol in workflow.md)

## Phase 5: Final Review and Archive

- [ ] Task: Run final conductor review
    - [ ] Review the track diff against the spec, plan, workflow, and tests
    - [ ] Apply any high-confidence fixes surfaced by review
    - [ ] Re-run validation until stable
- [ ] Task: Archive the completed Rust migration track
    - [ ] Move the track folder to the archive location
    - [ ] Update the tracks registry entry to completed
    - [ ] Preserve links to the inventory and evidence bundle
- [ ] Task: Conductor - Automated Review and Checkpoint 'Final Review and Archive' (Protocol in workflow.md)
