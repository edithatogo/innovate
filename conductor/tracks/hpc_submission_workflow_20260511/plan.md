# Implementation Plan: HPC Submission Workflow Arrangement and Registry Handoff

## Phase 1: Establish the HPC Execution Packet [checkpoint: pending]

- [x] Task: Confirm the blocked HPC targets and their evidence bundle
    - [x] Enumerate Spack, EasyBuild, HPSF, and E4S
    - [x] Confirm the candidate recipe, easyconfig, governance, and packet artifacts
    - [x] Identify the evidence logs and scheduler templates that already exist
- [x] Task: Write tests for packet and template presence
    - [x] Require the packet JSON to enumerate all blocked targets
    - [x] Require the scheduler and governance templates to exist
    - [x] Require the packet to stay non-claiming
- [x] Task: Conductor - Automated Review and Checkpoint 'Establish the HPC Execution Packet' (Protocol in workflow.md)

## Phase 2: Arrange Per-Target Execution Paths [checkpoint: b0552ec]

- [x] Task: Prepare Spack submission materials
    - [x] Fill in any remaining recipe metadata gaps
    - [x] Confirm the scheduler template captures the needed batch context
    - [x] Define the exact install and smoke commands to run
- [x] Task: Prepare EasyBuild submission materials
    - [x] Fill in any remaining easyconfig metadata gaps
    - [x] Confirm the scheduler template captures the needed module context
    - [x] Define the exact install and sanity commands to run
- [x] Task: Prepare HPSF governance materials
    - [x] Populate contacts, support policy, and maintenance cadence
    - [x] Define the scheduler-backed deployment evidence to attach
    - [x] Define the external review or registry handoff target
- [x] Task: Prepare E4S portability materials
    - [x] Populate CPU, GPU, and mixed-bridge evidence slots
    - [x] Define the accelerator metadata to capture
    - [x] Define the review or registry handoff target
- [x] Task: Conductor - Automated Review and Checkpoint 'Arrange Per-Target Execution Paths' (Protocol in workflow.md)

## Phase 3: Execute or Record External HPC Hand-offs [checkpoint: e9b0df2]

- [x] Task: Submit or run the Spack packet
    - [x] Execute the scheduler-backed recipe run or record the blocker
    - [x] Capture the batch log, install log, and smoke log
- [x] Task: Submit or run the EasyBuild packet
    - [x] Execute the scheduler-backed easyconfig run or record the blocker
    - [x] Capture the batch log, module sanity log, and smoke log
- [x] Task: Submit or hand off the HPSF packet
    - [x] Capture the governance review or registry contact evidence
    - [x] Record any blocker that prevents submission
- [x] Task: Submit or hand off the E4S packet
    - [x] Capture the portability review or registry contact evidence
    - [x] Record any blocker that prevents submission
- [x] Task: Capture HPC receipts and blocker notes
    - [x] Record registry URLs, review IDs, batch metadata, or note why a target remains blocked
    - [x] Preserve the resulting evidence in the repo
- [x] Task: Conductor - Automated Review and Checkpoint 'Execute or Record External HPC Hand-offs' (Protocol in workflow.md)

## Phase 4: Reconcile Docs, Packet, and Status Matrices

- [x] Task: Update the human-facing HPC documentation
    - [x] Reflect actual submission or blocker status in the readiness docs
    - [x] Link the per-target packet and evidence bundle
- [x] Task: Update the machine-readable HPC packet
    - [x] Refresh statuses and next-step notes for each target
    - [x] Add durable links to receipts, batch logs, or blocker notes
- [x] Task: Add regression tests for the reconciled state
    - [x] Test that the docs and packet point to the same target states
    - [x] Test that no page overstates a submission claim
    - [x] Test that all HPC packet artifacts remain present
- [x] Task: Conductor - Automated Review and Checkpoint 'Reconcile Docs, Packet, and Status Matrices' (Protocol in workflow.md)

## Phase 5: Final Review and Archive

- [ ] Task: Run final conductor review
    - [ ] Review the track diff against the spec, plan, workflow, and tests
    - [ ] Apply any high-confidence fixes surfaced by review
    - [ ] Re-run validation until stable
- [ ] Task: Archive the completed HPC workflow track
    - [ ] Move the track folder to the archive location
    - [ ] Update the tracks registry entry to completed
    - [ ] Preserve links to the packet and evidence bundle
- [ ] Task: Conductor - Automated Review and Checkpoint 'Final Review and Archive' (Protocol in workflow.md)
