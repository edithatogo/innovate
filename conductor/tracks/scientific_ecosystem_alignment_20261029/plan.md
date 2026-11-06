# Plan

- [x] Reconcile current source, active ownership and reusable evidence (archive reconciliation `5cd07b7f`; continuation-20261106.md).
- [x] Qualify Python 3.15.1 and PyArrow 26 base dependencies; optional Polars acceptance remains open.
- [~] Inventory actual Kairos calls and qualify the accepted upstream pin without copying a scheduler (inventory and retained-pin native examples pass; lock/integrator acceptance open).
- [~] Retain diffusion fitting and diagnostics here; qualify installed forecasts and adoption artifact handoff (selected installed Bass CPU/Arrow diagnostics pass; broader scientific acceptance open).
- [x] Write failing regression cases before implementation changes (`1a6e2752`: two unsupported-native assertions failed before the fix and passed afterward).
- [x] Run focused native and installed-consumer checks with resource receipts (`1a6e2752`: two retained-pin Rust examples and selected installed CPU/Arrow cases; scope limits retained).
- [~] Agent review, apply fixes and run required repository/hosted gates (local Conductor review and lint/types/docs/package pass; full unit and hosted gates open).
- [ ] Integrator acceptance and archive only after all native gates pass.

Ownership: planning writer 01a17093-d6f6-7302-a1fa-a868618ac648 in
/Volumes/PortableSSD/GitHub/.codex-worktrees/innovate-scientific-20261029. Existing repository integrator retains primary, shared records
and pins; this branch is a bounded proposal. No native operation is taken over.

## Current acceptance reconciliation

See [2 November reconciliation](reconciliation-20261102.md).

- [x] Publish compatible releases and verify the installed PyPI Arrow/EVPI case.
- [x] Verify authentication-only token issuance from main for both indexes.
- [ ] Complete the remaining native and optional-consumer gates before archival.

## Reconciliation lane — 6 November 2026

Chat `01a1b0dc-206d-7e31-9f61-544eb5cc16b5` owns archived-checkbox
reconciliation, the Kairos call inventory and bounded fail-closed evidence fixes
in `/Volumes/PortableSSD/GitHub/.codex-worktrees/innovate-reconcile-scientific-20261106`,
branch `codex/reconcile-scientific-20261106`, based on `b9998add`.
This is a standard Git worktree. The managed-tool request for the nested
repository was rejected; the parent request created an outer-wrapper checkout
without the project's files. That unused app attachment is retained because
the app refuses to archive its primary checkout.

The earlier planning assignment above remains historical context. Its separate
worktree and `.conductor/` state are preserved. No running native operation was
found. Scope: the three named archive plans and reconciliation records, this
track's records, Kairos adapter evidence and its tests/examples. Shared pins,
authentication, other repositories, releases and native tenant objects remain
with their existing owners. Next operation: reconcile seven archive checkboxes
against retained artifacts and notes, then test source-only Kairos claims before
attempting bounded native checks. Integration is a separate integrator action.

Terminal evidence: archive commit `5cd07b7f`, functional commit `1a6e2752`,
[continuation](continuation-20261106.md), [review](review-20261106.md), and
[required-gate receipt](full-gate-20261106.json). Current next operation is
integrator review of this isolated lane and exact-test runtime diagnosis; the
earlier “Next operation” entry above is retained as assignment history. No main
integration, pin promotion, public push, hosted run or release was performed.

## Rust and Polars redesign — user scope expansion, 6 November 2026

The same chat and isolated worktree now own the source migration proposed in
[migration-plan.md](migration-plan.md). The user authorises API redesign and
interim feature unavailability, while requiring every current and planned
scientific feature in the final Rust library. Python, Julia and R are the only
supported bindings; Rust is the core. Other language packages are removed.
This supersedes the earlier API-preservation and bounded-fix scope for this lane.
Main integration, external publication, native objects and other repository
writers remain outside the lane. No active local native operation was found.
Next operation: native-only extraction, capability gating and Polars contract
checks before changing installed package defaults.

## Native checkpoint — 7 November 2026

The expanded migration lane has delivered the Rust-only/Polars development
cutover and local installed Python, Julia, R and Voiage evidence. See
[native-cutover-20261107.json](native-cutover-20261107.json) and
[native-cutover-review-20261107.md](native-cutover-review-20261107.md).
The earlier 3.15.1 environment remains preserved; the active selected environment
is `.venv-native` running 3.15.2. All 38 scientific obligations still block release.
Next source operation: native multi-generation substitution and convergence/
identifiability reporting, followed by current Kairos dispatch qualification.
The branch is an isolated review packet; shared/main promotion stays with the
integrator. No public push, hosted dispatch, publication or authentication change
was performed. No active native test operation remains at the checkpoint.
