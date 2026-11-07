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

## Complementary goods and network/resource continuation

Same isolated writing lane resumes at 9a721b9, initially clean. Other writers
and native operations retain ownership. All existing/planned features remain required.
- [x] Port bounded complementary-goods equations through the native trajectory fitter. a07932b
- [x] Add static Kairos-owned network agent state and FIFO resource service. a07932b
- [x] Validate independent limits, conservation and Python/Julia/R consumers. a07932b
- [x] Retain evidence and remaining historical/scientific/platform obligations. a07932b

Evidence: network-resource-20261107.json; review: network-resource-review-20261107.md.
57 Rust tests, 27 installed Python checks, 142 Julia assertions and R cases pass.
Core/C ABI line coverage: 81.26%. Eight bounded native families; eight families
remain unavailable, and all 38 full-feature obligations remain required. The
Conductor goal stays active. Next source work: remaining historical models
(including lock-in), covariate/event extensions and advanced inference/scenarios,
then richer graph/resource behavior and outstanding distribution qualification.

Source/evidence commit: `a07932b5bf2e845c4f4d217ade2b5899b97bb351`. No native operation remains in flight.

## Lock-in continuation — in progress

Same writing lane resumes at 9ca4088; initial checkout was clean. Previous goal
turn was progress (committed source and native evidence). This turn recovered
LockInModel equations from 56d91dd7, generated an independent SciPy DOP853 fixture
on selected Python 3.15.2, and confirmed the new regression fails specifically
because native lock_in is unavailable. Owned uncommitted paths: tests/lock_in.rs
and tests/fixtures/lock_in_scipy.json under innovate-core, plus this track's record.

- [x] Port bounded lock-in with separate historical RHS-state and output clipping.
- [x] Qualify bounded fitting, rates/forecast semantics and three binding consumers.

Red receipt: evidence/lock-in-in-progress/red.log. This is intentional TDD red
state, not a passing native checkpoint. The model remains unavailable in the
ledger/discovery; no feature or release acceptance changed. No native operation
remains live. Next action: implement lock-in in the native trajectory engine,
then run the exact regression before expanding validation. Goal remains active.

Lock-in implementation progress: native dispatch, parameter validation,
historical RHS clipping and output clipping are implemented. Three focused
regressions pass: independent DOP853 trajectories, output clipping and fitted
trajectory recovery with an explicit interior structural-identifiability warning.
Free market potential is confounded with feedback/competition coefficients in
interior trajectories; boundary information is not qualified by these tests.
Discovery/ledger acceptance remains unchanged pending binding and broader gates.
Owned source edits are competition.rs, kernel.rs and validation.rs plus the
lock-in tests/fixture and this record. No native operation remains live. Next:
forecast/rate semantics, installed Python/Julia/R validation and evidence review.

Lock-in binding progress: the four focused Rust regressions pass, including an
independent held-out decoupled logistic limit and explicit historical rate
labeling. Rebuilt/installed Python wheel: 28 native tests pass. R native cases
and Julia's 144 assertions pass, including the lock-in analytic forecast. These
are local bounded results; discovery/ledger promotion and full current-head
coverage/review/packaging gates remain next. Owned edits also include the three
binding test files. No native mutation or external publication occurred.

Lock-in discovery now advertises the ninth bounded family. Current rebuilt
artifacts pass 61 Rust tests (57 core/C ABI coverage-run plus 4 Polars), 28
installed Python checks, 144 Julia assertions and R native cases. Workspace
Clippy and package metadata pass. Core/C ABI line coverage: 81.43%. Development
policy still reports all 38 full-feature obligations as release blockers.
Receipt lock-in-20261107.json is explicitly in-progress, not final acceptance.
Next review fix: preserve the historical fitting bound m >= max(observed) and
its max-series initializer; then finish documentation/static/evidence checks
and commit this owned packet. Existing tests/fixtures remain the oracle. No
native operation remains live; the Conductor goal remains active.

Market-bound review fix: a new regression reproduced acceptance of an initial m
below the second series' maximum. Lock-in now rejects that seed, transforms m as
max(observed)+positive offset during optimization, and initializes m at 1.5 times
the maximum across both series. Five focused lock-in regressions and workspace
Clippy pass. Receipts: evidence/lock-in-in-progress/bound-{red,green}.log.
Earlier full coverage/artifact totals precede this fix and are not current-head
acceptance. Rebuild is running in exec session 46008; coverage in session 73818.
These are verified live handles, not inferred waits. Next action: poll these
handles, install/test the resulting wheel and C ABI consumers, then finalize
review/evidence and commit. Do not restart merely because observation times out.

Lock-in local checkpoint: 62 Rust tests, 28 installed Python checks, 144 Julia
assertions and R post-bound native cases pass. Core/C ABI coverage: 81.48%.
Review: lock-in-review-20261107.md. Fresh postbound logs supersede the disk-failed
attempt; previous stale logs are not final acceptance. No native operation remains
live. Nine bounded families; seven remain unavailable. Next: remaining historical
model ports, covariates/events, inference/scenarios and full distribution gates.
Goal remains active; all 38 obligations remain required.

Lock-in source/evidence commit: `37300334bf7659aa68d1ed30d7f7f60e4922b1ae`. All operations terminal; next source tasks remain above.

## Historical network-diffusion continuation — in progress

Same writer resumes at 3b0ccdfb; initial checkout clean. Previous turn was progress
through committed lock-in validation. Recovered models/network.py and
NetworkDiffusionInputs from 56d91dd7. This family fits node-level base curves,
uses row-normalized adjacency spillover, clips negatives and applies cumulative
max over time. It is distinct from the Kairos stochastic event engine.

- [x] Port bounded weighted spillover prediction, isolated-node behavior and matrix output.
- [x] Port bounded node fitting/state, diagnostics/summary and historical rate conventions.
- [x] Validate bounded Python/Julia/R consumers and retain evidence before advertising capability.

Owned new source: rust/crates/innovate-core/tests/network_diffusion.rs.
Independent closed-form Bass predictions define expected weighted node outputs.
The exact test fails because network_diffusion remains unavailable. Red receipt:
evidence/network-diffusion-in-progress/red.log. No release/capability claim or
ledger qualification changed. No native operation remains live. Next action:
implement the native network wrapper and pass this regression, then port fitting
and summary/diagnostic semantics without equating this model to the event engine.
All full-feature obligations and the Conductor goal remain active.

Native historical network wrapper implemented in spillover.rs. Three focused
regressions pass: weighted/isolated-node analytic predictions, independent node
fitting with reusable state and held-out forecasts, and invalid graph/options.
Core all-targets Clippy passes. Output is always time_node; inputs also accept
node_time or explicit historical_auto. Spillover strength is held fixed and
node fits ignore spillover, matching the historical algorithm; diagnostics warn
that adjusted training predictions need not match observations. Historical
sample increments are explicitly not time derivatives. Unavailable covariate,
stochastic and unsupported base-model paths fail rather than falling back.

Capability remains unadvertised pending broader review/consumer checks. Next:
complete historical summary/coordinates and unsupported-feature accounting,
then validate installed Python/Julia/R and current-source coverage. Targeted
interventions, full base-model composition, covariates and stochastic uncertainty
remain required; the old intervention setter did not actually change predict.
Owned source edits include spillover.rs, lib.rs, kernel.rs and network tests.
No native operation remains live; full goal remains active.

Historical summary now retains coordinates, edge count/density and weighted
average degree, strips adjacency diagonals like the original constructor, and
supports no-time summaries without fabricating predictions. Four focused Rust
regressions pass. Polars node fitting accepts its normal empty fit_options;
nonempty unsupported fit options still fail explicitly. Rebuilt/installed
consumers pass 29 Python tests, 146 Julia assertions and R conformance including
weighted-network prediction. Workspace all-target Clippy passes. New sources
remain owned by this lane and uncommitted; capability not yet advertised.

Next: review finite graph-statistic arithmetic/unsupported feature handling,
then discovery/ledger, current-source coverage, documentation/package checks,
final evidence and commit. Coordinates and summary fields are covered by the
new native regression. No native operation remains live; goal stays active.

Network review found graph-average overflow from summing finite row degrees
before division; a new regression failed before the stable-mean fix and now
passes. Unknown payload options now fail explicitly rather than silently
ignoring stochastic/intervention requests. Five focused network regressions
pass. Discovery advertises the tenth bounded family with deterministic spillover
semantics and no derivative-rate claim. Ledger receipt remains in-progress.
Current-source coverage run: 63 core/C ABI tests pass; line coverage 81.72%.
Workspace Clippy passes. Rebuilt installed artifacts pass 29 Python tests,
146 Julia assertions and R conformance, including the new discovery count.
No native operation remains live. Next: Polars native gate, docs/package/static
checks, final immutable evidence/review and commit; then remaining six model
families and all outstanding full-feature qualifications. Goal remains active.

Network diffusion checkpoint: 67 Rust tests, 29 installed Python checks, 146 Julia
assertions and R native cases pass. Core/C ABI line coverage: 81.72%. Review:
network-diffusion-review-20261107.md; receipt: network-diffusion-20261107.json.
Ten bounded families; six remain unavailable. All 38 full-feature obligations
remain required. Next: policy/latent/regime and hierarchical/mixture/composite
ports, remaining covariate/event/inference work and distribution qualification.
No native operation remains live; full Conductor goal remains active.

Historical network source/evidence commit: `975248c8dd7bac64651eb5d609556d9fd9cba509`. All operations terminal; remaining source tasks stay active.

## Historical policy-hazard continuation — in progress

Same writer resumes at 99da4e78; initial checkout clean. Previous goal turn made
progress through committed historical network qualification. Recovered policy.py
and PolicyTimingInputs from 56d91dd7. policy.rs now reuses native scalar curve
fit/predict operations and preserves fixed decay, independently fitted baseline,
decaying event level multipliers, nonnegative clipping and cumulative maxima.
This is not a calibrated survival hazard or causal effect; metadata says so.

- [x] Port bounded historical policy prediction/fit/state, summaries and sample increments.
- [x] Validate event edge cases and bounded Python/Julia/R consumers before discovery.
- [x] Complete local coverage/review/docs/package evidence and retain remaining scope.

Independent closed-form boundary/decay regression initially failed with native
unavailability and now passes. Additional tests exercise fixed-decay baseline
fitting and forecasts, no-time summary, negative multiplier clipping and malformed
inputs. Source edits: policy.rs, lib.rs, kernel.rs and tests/policy.rs. Capability
remains unadvertised and ledger unchanged pending broader validation. Next:
inspect the exact focused test/Clippy results, add singleton event-array transport
for R, then installed binding and current-source qualification. Full base-model
composition/covariates, policy parameter inference, stochastic draws and causal
validation remain required. Goal remains active.
Focused results inspected: all four policy regressions and core all-target
Clippy pass. No native operation remains live. Next action is binding transport
and installed-consumer validation, not rerunning unchanged focused checks.

Policy consumers now pass 30 installed Python tests, 147 Julia assertions and
R native conformance including singleton event arrays and exact event-boundary
prediction. Polars baseline fitting/forecast state also passes. R known-array
handling now includes event times/effects/labels and node/covariate labels,
preserving input maps through the existing separate covariate-map branch.
Current-source core/C ABI coverage and workspace Clippy completed successfully;
artifact rebuild and tests are terminal. Capability remains unadvertised pending
final review, discovery/ledger, docs/package and evidence gates. Next action is
inspect coverage totals, finish these gates and commit this bounded port. Full
policy inference/covariates/stochastic and causal qualification remain required.

Policy checkpoint: 71 Rust tests, 30 installed Python checks, 147 Julia assertions
and R cases pass. Core/C ABI coverage: 81.99%. Review: policy-review-20261107.md;
receipt: policy-20261107.json. Eleven bounded families; five remain unavailable.
All 38 obligations remain required. Next: latent/regime, hierarchical/mixture/
composite ports, inference/scenarios, remaining covariates/events and distribution
qualification. No native operation remains live; full goal remains active.

Policy source/evidence commit: `34a85b209efc7b2187f0948b0f319bfd07701907`. All operations terminal; remaining source tasks stay active.

## Heterogeneous Kairos network/resource continuation

Same writing lane resumes at 20b0f667 in the verified isolated worktree; main
and other worktrees retain their owners. No native operation is active.

- [x] Add explicit per-agent innovation/imitation multipliers and service ticks,
  retaining homogeneous defaults, Kairos FIFO ownership and deterministic replay.
- [x] Validate exact heterogeneous queue behavior in Rust, Python, Julia and R.
- [x] Retain source/review evidence and remaining model/feature obligations.

Historical latent-process source inspection confirms exponential residual
smoothing, linear interpolation with endpoint continuation, clipping and batch
cumulative maxima. Its probabilistic inference and the five remaining model
families remain required; no capability is retired by this network extension.

Heterogeneous network source/evidence commit: `6a606f5b1005e160378a5593de22886fae30f2a6`.
74 Rust, 31 installed Python, 152 Julia assertions and R conformance pass;
82.09% core/C ABI line coverage. Clippy, lint/types, docs (163 pages), local
wheel install and archive metadata pass. Receipt: heterogeneous-20261107.json;
review: heterogeneous-review-20261107.md. No native operation remains active.
Next source task: historical latent-process smoothing/state/interpolation port,
then regime/hierarchical/mixture/composite and richer graphs/resources. All 38
obligations remain required, release guard still fails intentionally, goal active.

## Historical latent-process continuation

Same writing lane resumes at 3773d3f6, clean isolated checkout; no live native
operation. Previous goal turn made committed, verified Kairos progress.

- [x] Port historical latent residual smoothing, endpoint interpolation, state,
  forecasts, diagnostics, summaries and sample increments without statistical overclaims.
- [x] Qualify installed Python/Julia/R transport including saved latent arrays.
- [x] Review, record source-bound evidence and retain every remaining obligation.

Latent source/evidence commit: `bfce5a1830581a2dd341d5a57105f7a0bc60f3fe`.
79 Rust tests, 32 installed Python, 155 Julia assertions and R pass. Core/C ABI
line coverage 82.83%, latent module 91.75%; Clippy, lint/types, docs and package
metadata pass. Receipt/review: latent-20261107.json and latent-review-20261107.md.
All operations terminal. Twelve bounded native families, four unavailable; all
38 obligations remain required and release guard fails intentionally.

Next: regime-switching. Preserved advanced.py uses a detected split index applied
to each requested prediction batch, plus a continuity offset between regimes;
short batches can therefore use only the first regime. Qualify that historical
variant explicitly and consider a separately named fixed-time variant for
consistent forecasts; do not silently erase historical semantics. Automatic
changepoint detection and both segment estimators need native qualification.
Hierarchical/mixture/composite, uncertainty/inference, covariates/events, richer
Kairos graph/resource behavior and distribution qualification remain required.
Goal remains active; no main integration, push or hosted acceptance asserted.
