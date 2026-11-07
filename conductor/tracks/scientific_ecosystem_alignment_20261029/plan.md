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

## Continued native feature migration — 7 November 2026

User instruction resumes the same writing lane at 2e0b9930. Checkout is clean;
other worktrees and native operations retain their owners. This chat owns Rust
scientific modules, thin bindings, affected tests/docs and this track's records
on codex/reconcile-scientific-20261106. No main or shared environment promotion.

- [x] Port bounded multi-generation Norton–Bass prediction, rates, fit and diagnostics; explicit historical equations and independent oracles pass (native-features-20261107.json). 5b0df5f
- [x] Add native optimizer convergence and local identifiability reporting to all seven fitters; local-rank and non-convergence cases pass. 5b0df5f
- [x] Extend Polars multi-output fitting/prediction and validate installed Python, Julia and R consumers (local macOS; packaging/platform limits remain). 5b0df5f
- [~] Continue remaining model/feature obligations: bounded Lotka–Volterra and multi-product competition now pass independent ODE/analytic and binding cases; current Kairos qualification is next.
- [x] Review this bounded implementation, update capability/feature evidence and preserve failing full-release gates (native-features-review-20261107.md). 5b0df5f

The established native workflow uses repo-local Python 3.15.2 and focused Cargo/
pytest selectors. Earlier Python 3.14 workflow examples are superseded by the
current technology decision and user runtime instruction.

The continuation receipt records 37 Rust tests (including the coverage run),
22 installed Python checks, 128 Julia assertions and the R native cases. Overall
core/C ABI coverage remains below 80%; all 38 full-feature obligations remain
release blockers. The next source task is current Kairos execution integration,
then remaining covariate/model/inference obligations. A bounded read on 7 November
found remote Kairos HEAD `d56863505d938feb069af88a737eb7668113d14b` while its
separate local checkout was at `339aba155cc5ea7d393a5bd78a7947eddae26e03`. Neither
checkout nor Innovate's dependency pins were changed by this continuation.

Bounded source/evidence commit: `5b0df5fdf0bfddf540dc438264d57b54261966da`. The track remains active;
current ownership stays with this isolated lane. No native operation remains
in flight at this checkpoint. Next work is the uncompleted source tasks above.

## Kairos native continuation

Same chat and writing checkout resume at ab7b3a3a; checkout is clean. Ownership
remains Rust modules, bindings/tests and this track. Remote Kairos HEAD readback
is d56863505d938feb069af88a737eb7668113d14b. Other checkouts remain untouched.

- [x] Connect Kairos scheduling/RNG to a bounded native finite-population Bass birth process. 5953b38
- [x] Validate replay, analytic limits, sampling invariance and invalid options. 5953b38
- [x] Validate installed Python, Julia and R consumers; update evidence and ledger (kairos-integration-20261107.json). 5953b38
- [ ] Continue ABM, covariates and remaining full-feature obligations.

Local continuation: 41 Rust tests, 23 installed Python checks, 131 Julia assertions
and R native conformance pass. Review: kairos-review-20261107.md. Full migration
remains active. Next source task: ABM agent/state integration and covariate/event
ports, retaining all remaining model/inference obligations. Disk exhaustion was
resolved for these checks by removing owned build caches; storage remains tight.

Source/evidence commit: `5953b385cea165e8e0f327f27ebed920004b002d`. No native operation remains in flight.

## ABM and covariate/event continuation

Same writing lane resumes at 3cc10543 with a clean checkout. Other checkouts and
native owners remain untouched. Scope: native ABM state, covariate/event ports,
three binding consumers and this track's records; all existing obligations remain.

- [x] Integrate Kairos agent identities/components with homogeneous adoption events. 897ed8e
- [x] Validate agent conservation, replay and DES-count equivalence. 897ed8e
- [~] Port covariate/event scientific semantics: simulation-only additive forcing and independent binomial oracles pass; historical interpolation/fitting and event-split curves remain.
- [x] Validate Python/Julia/R for this bounded port; retain evidence and reconcile remaining work. 897ed8e

Evidence: abm-forcing-20261107.json; review: abm-forcing-review-20261107.md.
45 Rust tests, 24 installed Python checks, 135 Julia assertions and R cases pass.
Core/C ABI line coverage is 80.18%; full scientific/platform gates remain open.
One active track remains in both the registry and active track directories.
Next source work: historical covariate interpolation/fitting and event-split
curves, heterogeneous/network ABM, then remaining model/inference obligations.

Source/evidence commit: `897ed8e16da4a00e0a9907b11c0b617fc2a80bad`. No native operation remains in flight.

## Historical model continuation

Same writer/worktree resumes at 84eca121, initially clean. No ownership change.
- [x] Port bounded Logistic additive covariates and historical independent event curves, ec21725
      with explicit handling of the old event path ignoring covariates.
- [x] Qualify explicit-initialized fitting, forecasts and Python/Julia/R consumers. ec21725
- [ ] Continue other historical models and richer/network ABM; preserve all obligations.

Initial-adopter ABM states are also implemented and tested. Evidence:
historical-logistic-20261107.json; review: historical-logistic-review-20261107.md.
51 Rust tests, 25 installed Python checks, 137 Julia assertions and R native
cases pass. Core/C ABI line coverage: 80.75%. All 38 obligations remain required.
Next: other historical model covariates/event splits, advanced initialization
and rate helpers, heterogeneous/network/resource ABM, and remaining model/inference
ports. Full scientific, platform and release qualification remain open.

Source/evidence commit: `ec217253756eeed2e71e9f665ef943dc3472bffd`. No native operation remains in flight.
