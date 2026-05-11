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

## Phase 2: Promote or Explicitly Reassign Remaining Slices

- [ ] Task: Implement native Rust execution for promotable slices
    - [ ] Promote remaining core operations that can be made native
    - [ ] Promote remaining diffusion-family slices that can be expressed natively
    - [ ] Promote remaining competition-family slices that can be expressed natively
    - [ ] Promote remaining payload and schema reconciliation slices that can be expressed natively
    - [ ] Remove undocumented bridge fallback for promoted slices
- [ ] Task: Record explicit non-Rust ownership where native promotion is not appropriate
    - [ ] Preserve Python-reference ownership only where the contract requires it
    - [ ] Document any families that remain explicitly promoted elsewhere
    - [ ] Keep bridge fallback limited to documented non-native paths
- [ ] Task: Conductor - Automated Review and Checkpoint 'Promote or Explicitly Reassign Remaining Slices' (Protocol in workflow.md)

## Phase 3: Prove Parity, Profiling, and Binding Stability

- [ ] Task: Add and update parity and regression tests
    - [ ] Test promoted slices against Python reference semantics
    - [ ] Test error mapping and unsupported-payload behavior
    - [ ] Test fallback routing for explicitly non-native families
- [ ] Task: Build a binding smoke matrix for each promoted family
    - [ ] Cover Rust and Python for every promoted slice
    - [ ] Cover applicable downstream bindings: R, Julia, TypeScript, Go, and C#
    - [ ] Record the matrix in the evidence bundle and ledger
- [ ] Task: Capture performance and profiling evidence
    - [ ] Capture benchmark output for promoted slices
    - [ ] Capture CPU profiling and memory profiling where relevant
    - [ ] Capture any XLA or accelerator evidence that applies
    - [ ] Attach family-level profiling evidence to the ledger or evidence bundle
- [ ] Task: Validate all binding surfaces
    - [ ] Run Rust binding smoke coverage
    - [ ] Run Python, R, Julia, TypeScript, Go, and C# smoke checks as applicable
    - [ ] Preserve the resulting evidence bundle in the repo
- [ ] Task: Conductor - Automated Review and Checkpoint 'Prove Parity, Profiling, and Binding Stability' (Protocol in workflow.md)

## Phase 4: Reconcile Docs, Inventory, and Claim Language

- [ ] Task: Update the Rust roadmap and related docs
    - [ ] Reflect the final ownership boundary in the roadmap text
    - [ ] Update architecture or binding docs if claim language changes
    - [ ] Remove any roadmap wording that overstates bridge ownership
- [ ] Task: Reconcile claim language across docs and inventory
    - [ ] Remove "partial" or equivalent wording once the ledger says full ownership is closed
    - [ ] Keep the archived closure track available as evidence, not as the active source of truth
- [ ] Task: Refresh the machine-readable migration inventory
    - [ ] Set terminal ownership for every slice
    - [ ] Record promotion blockers and rationale where needed
    - [ ] Keep the docs and inventory synchronized
- [ ] Task: Add regression tests for the reconciled state
    - [ ] Test that the roadmap matches the inventory
    - [ ] Test that no page claims full Rust ownership without evidence
    - [ ] Test that the inventory and docs remain aligned
- [ ] Task: Conductor - Automated Review and Checkpoint 'Reconcile Docs, Inventory, and Claim Language' (Protocol in workflow.md)

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
