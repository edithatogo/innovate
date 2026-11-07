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

## Regime-switching continuation

Same writer resumes at 9dfd1123 in the clean isolated worktree; no active native
operation. Prior goal turn made committed latent-process progress.

- [x] Port two-regime fitting and explicit historical batch-index prediction;
  qualify native penalized L2 grid changepoint detection and fixed-time forecasts.
- [x] Validate saved regime state through Python, Julia and R.
- [x] Complete local review, coverage, docs/package evidence and ledger update.

Historical detection uses ruptures Pelt L2, min_size=2, jump=5, penalty=3;
the wrapper takes the first detected boundary or falls back to the midpoint.
The native implementation will independently minimize the same grid objective
without copying upstream source; detector algorithm/tie semantics stay explicit.
Every remaining model/feature obligation remains required.

Regime source/evidence commit: `529d8df06720a16ae4af880e33d3d90cc3623672`.
86 Rust tests, 33 installed Python, 159 Julia assertions and R conformance pass.
Core/C ABI line coverage 83.22%; regime module 89.02%. Clippy, lint/types,
docs (163 pages) and local package metadata/source inclusion pass. Receipt and
review: regime-20261107.json and regime-review-20261107.md. Native grid detector
qualifies seven external reference fixtures; arbitrary ties/performance, other
objectives/search methods and full scientific acceptance remain open.

Explicit fit-time boundary ignoring was caught and fixed; exact times between
samples persist. Identified two-point Fisher–Pry fitting now preserves the
historical short-regime path, with no residual degrees of freedom and no AIC/BIC.
No rank or unsupported-family gates were removed. All operations are terminal.
Thirteen bounded native families; hierarchical, mixture and composite remain
unavailable. All 38 obligations remain required; release guard still fails.

Next: hierarchical wrapper, preserving the historical distinction between
group-series fits and shared-series fits, global/group parameter aggregation
and group forecasts, while qualifying statistical limitations explicitly.
Then mixture/composite, inference/uncertainty/scenarios, remaining covariates
and richer Kairos graphs/resources, distribution and scientific qualification.
Full goal remains active; shared main and other writing lanes remain untouched.

## Hierarchical continuation

Same writer resumes at 59efbae0, clean isolated checkout, no live native operation.
The previous turn made committed regime-switching progress.

- [x] Port shared/group-series fitting, global/group parameter aggregation and
  forecasts; preserve historical additive behavior and explicitly qualify a centered variant.
- [x] Validate aggregate/group Polars forecasts and saved Python/Julia/R state.
- [x] Complete local coverage/review/docs/package evidence and ledger update.

Preserved hierarchical.py fits each group independently, averages parameters
into global values, then adds those globals to each absolute group estimate at
prediction time. Shared observations instead fit once and set group adjustments
to zero. Preserve that historical distinction without calling it partial pooling
or Bayesian inference; an explicit centered parameterization will recover the
independently fitted group curves. All remaining obligations stay required.

Hierarchical source/evidence commit: `73146b8bea10eb0edbe611ef7fe4881c2fecbe9e`.
92 Rust tests, 34 installed Python, 162 Julia assertions and R conformance pass.
Core/C ABI line coverage 83.41%, hierarchical module 86.28%. Clippy, lint/types,
docs (163 pages), wheel installation and archive metadata/source inclusion pass.
Receipt/review: hierarchical-20261107.json and hierarchical-review-20261107.md.
Centered is the redesigned API default; historical_additive is explicit and
retained. A failing large-constant R² test drove stable mean/undefined-variation
handling in shared diagnostics. No native operation remains active.

Fourteen bounded families; mixture and composite remain unavailable. All 38
obligations remain required, optimized release guard intentionally fails. Next:
mixture source uses an adoption-level responsibility heuristic with weighted
component fits and normalized weights. Port weighted scalar least squares and
qualify that historical variant explicitly; do not present adoption-level
responsibilities as a calibrated latent-class likelihood. Gaussian likelihood
mixtures and complete inference remain required. Preserve component-rate helpers,
simulation/summary behavior and model/weight state in the native contract.
Composite, remaining inference/uncertainty/covariates/scenarios, richer Kairos
graphs/resources and distribution/scientific qualification remain required.
Goal active; shared main and other writing lanes untouched.

## Mixture and weighted-fitting continuation

Same writer resumes at 98b5bba1, clean isolated checkout, no live native operation.
The prior goal turn made committed hierarchical progress.

- [x] Add native scalar weighted least squares with finite nonnegative precision
  weights, explicit effective sample/rank diagnostics and independent reference checks.
- [x] Port mixture prediction/state plus historical adoption-responsibility updates;
  qualify a residual-likelihood variant without claiming calibrated class inference.
- [x] Validate Python/Julia/R, review, coverage, docs/package and retained ledger scope.

Historical ScipyFitter curve_fit converts weights to sigma=1/sqrt(weights);
other old optimizer branches may ignore sigma. Native weighting will use explicit
sum-of-weighted-squared-residuals semantics consistently. No failure may silently
become successful EM convergence; retain per-component update diagnostics. All
38 scientific obligations remain required.

Mixture/weighting source-evidence commit: `ff20e4c69e0b85f0ad1ef6366344b1cd86d1e9c0`.
101 Rust tests, 36 installed Python, 166 Julia assertions and R conformance pass.
Core/C ABI line coverage 84.13%; mixture 88.67%, scalar fitting 92.88%.
Clippy, lint/types, docs (163 pages), local wheel installation and archive
metadata/source inclusion pass. Receipt/review: mixture-20261107.json and
mixture-review-20261107.md. Native Gaussian EM recovers separated components
with and without a warm start. Failed updates roll back, remain recorded and
cannot report convergence. Historical adoption responsibility mode is explicit.
No native operation remains active.

Fifteen bounded model families; composite remains unavailable. All 38 obligations
remain required and the optimized release guard intentionally fails. Next:
composite scalar-product ODEs, retaining interactions and parameter suffixes,
qualifying initial-condition/origin semantics and matrix forecasts via all three
bindings. Preserved composite.py resets zero initial state at each requested
batch start; preserve that as an explicit historical option while supporting
fixed-origin/stateful forecasts. Then cross-model rate/covariate/event coverage,
inference/uncertainty/scenarios, richer Kairos networks/resources, platform and
scientific qualification. Model-family availability never closes full-feature
acceptance. Shared main unchanged, goal active.

## Composite continuation

Same writer resumes at 95766f22, clean isolated checkout; no active native operation.
Prior turn made committed weighted-fitting/mixture progress.

- [x] Port heterogeneous scalar growth ODEs, signed interactions, fixed-origin
  forecasts and explicit historical batch resets, with matrix levels/rates and fitting.
- [x] Qualify independent ODE/analytic references and Python/Julia/R consumers.
- [x] Complete coverage/review/docs/package evidence and retain full feature scope.

Continuation found shared weighted-option parsing also reached competitive and
historical covariate/event fitters that do not apply weights. A new failing
regression confirms the gap; scalar_options now alone admits weights, while
shared options reject them explicitly. No weighted feature is silently ignored.
Composite preserves growth_i - sum(alpha_i_j * state_j), signed off-diagonal
coefficients and parameter suffixes. Logistic x0, Gompertz b and Fisher-Pry t0
are absent from these ODEs; do not claim they can be inferred from trajectories.

Composite source/evidence commit: `8e7fd9d35e797433611acddf698ba9c77fdbb8b7`.
110 broad Rust tests plus one focused capacity-policy test pass; 37 installed
Python, 170 Julia assertions and R conformance pass. Core/C ABI line coverage
84.32%, composite 87.00%. Clippy, lint/types, docs (163 pages), wheel install
and fresh archive metadata/exact source inclusion pass. Review/receipt:
composite-review-20261107.md and composite-20261107.json. Capacity test was
added after the broad run; unchanged production source/results were reused,
with no additional focused coverage claimed. Shared weighting dispatch gap is
fixed, with unsupported paths explicitly rejecting weights. All operations terminal.

Final package output redirection hit internal-volume exhaustion. Earlier package
logs were not treated as fresh proof; the current archive and package logs are
under target/composite-final-artifacts and target/composite-final-evidence on the
owned external volume, copied/hashes retained in track evidence. Prefer this
external artifact/log route for subsequent work; no other client cache was removed.

All sixteen model families now have bounded native support, with all 38 original/
planned obligations still required and release guard intentionally failing. This
is not full scientific or distribution acceptance. Before the next source wave,
reconcile active Conductor guideline/workflow Python 3.14 and NumPy/JAX profiling
references with the approved Rust/Polars/Python 3.15.2 design; preserve historical
evidence and any explicit backend exceptions. Then extend historical Bass/Gompertz
ODE covariates/events with fixed-history forecasts (their old prediction paths
are not the pointwise Logistic contract), followed by richer Kairos graphs/resources
and cross-model rate coverage, inference/uncertainty, scenario/data/reporting,
platform/clean-source/stiff-solver and scientific qualification. Goal active;
shared main and other writing lanes unchanged.

## Active instruction reconciliation

Same writer resumes at f1828333 in the verified isolated worktree; no native
operation is active. Previous turn made committed composite/dispatch progress.

- [x] Reconcile active product/workflow/style guidance and current-scope docs with
  Rust/Polars, selected Python 3.15.2 and actual native test/typing surfaces.
- [x] Verify runtime/manifest agreement, links, focused documentation hooks and
  retain unfulfilled strict typing, mutation, module coverage and release obligations.

The tech-stack pre-migration baseline is explicitly historical and is preserved.
Product context already states the user-authorized architecture. Active guidance
and README/native-scope introduction contain stale baseline/path claims. This
corrects current operating instructions, without altering historical evidence,
branding, user approvals, acceptance thresholds or the feature ledger.

Instruction source/evidence commit: `f101fed16b023ce57a2f2684f8b63de4ca16d584`.
Selected Python 3.15.2, runtime/manifest/configuration facts, nox sessions, local
links, hooks and 163-page Astro build pass. No numerical source changed; composite
8e7fd9d3 receipts are reused without a new coverage/package acceptance claim.
Review/receipt: instruction-review-20261107.md and instruction-20261107.json.
Strict public typing, branch/native mutation qualification, critical-module
coverage targets and all 38 release obligations stay open. Historical stack
preserved; current guidance no longer prescribes Python 3.14/NumPy/JAX execution.
All operations terminal; actual source worktree and ownership unchanged.

Next source task: Bass/Gompertz mechanistic covariates/events with saved forcing
history, fixed-origin forecasts and explicit historical batch-origin options.
Preserved Bass predicts via ODE from zero; Gompertz uses an ODE from 1e-6 with
batch cumulative maxima. These are distinct from pointwise Logistic covariates.
Qualify linear interpolation, exact event boundaries, conditional fit/state and
independent references; record legacy coefficient/interpolation defects rather
than silently treating a new equation as source parity. Continue richer Kairos
network/resource behavior, rates, inference and distribution/scientific gates.
Full goal remains active; no main integration, push or release performed.

## Mechanistic Bass/Gompertz forcing continuation

- [x] Add declared ODE covariates, event segmentation, saved forcing history and
  fixed-origin forecasts, preserving explicit historical-origin/layout options.
- [x] Qualify analytic/independent references, conditional fitting and Python/Julia/R.
- [x] Review and retain local evidence, solver and full-feature limitations.

Same writer/worktree continues at 6afb62c7. No native operation was live before
this source change. Default corrected coefficient layout uses shared named beta
coefficients in both event regimes. Historical layout remains explicit because
old pre-event indexing could read post parameters as coefficients. Gompertz's old
scalar interpolation-grid bug is not promoted as working covariate support.

Source: 44e56a76. Local qualification: 120 Rust tests, 38 Python tests,
175 Julia assertions and R conformance. All 38 release obligations remain.

## Directed and weighted Kairos resource network

- [x] Support directed and weighted exposure while preserving legacy replay.
- [x] Qualify direction, numerical scaling, FIFO and the three bindings.

Source: f78f1217. Eight focused Rust network tests, 39 installed Python tests,
177 Julia assertions and R conformance passed. Scheduled graph/resource
changes and remaining historical forcing are still open.

## Scheduled Kairos contacts and service duration

- [x] Implement validated graph replacements and service duration overrides.
- [x] Verify event ordering and the installed Python, Julia and R consumers.

Source: 55c6331c. Nine focused Rust network tests, 39 Python
tests, 177 Julia assertions and R conformance passed. Capacity changes, priority,
abandonment, forcing interoperability and historical model extensions remain open.

## Kairos resource priorities

- [x] Integrate native priority service with FIFO ties and no preemption.
- [x] Verify queue order and Python, Julia and R payloads.

Source: 947546ab. Ten focused Rust tests, 39 installed Python
tests, 178 Julia assertions and R conformance passed. Abandonment and capacity
changes remain open. The pinned Kairos RenegingQueue is separate from
PriorityResource and requires an explicit integration for wait expiry.

## Kairos waiting deadlines

- [x] Integrate native waiting expiry with priority service and terminal withdrawal.
- [x] Verify deadline ties, resource accounting and Python, Julia and R traces.

Source: ba7fe987. Eleven focused Rust tests, 40 installed
Python tests, 180 Julia assertions and R conformance passed. Scheduled capacity,
balking, optional retries and forcing interoperability remain open.

## Scheduled Kairos service capacity

- [x] Admit priority waiters on increases and preserve active services on decreases.
- [x] Verify pause/reopen, resource accounting and Python, Julia and R payloads.

Source: 265d36ef. Twelve focused Rust tests, 40 installed
Python tests, 181 Julia assertions and R conformance passed. Balking, optional
retries, historical model forcing and release qualification remain open.

## Kairos queue entry and retry policies

- [x] Bound waiting queues and restore susceptibility after optional cooldown.
- [x] Verify balking, expiry, retry exhaustion and Python, Julia and R traces.

Source: 769602e5. Thirteen focused Rust tests, 41 installed
Python tests, 184 Julia assertions and R conformance passed. Forcing
interoperability, historical model extensions and release qualification remain open.

## Kairos piecewise constant hazard interventions

- [x] Replace Bass p/q at schedule boundaries while preserving in-flight service.
- [x] Verify zero-hazard wakeup, shutoff and cross-binding schedule behavior.

Source: 040ed666. Fourteen focused Rust tests, 41 installed
Python tests, 185 Julia assertions and R conformance passed. Continuous forcing
interoperability, historical model extensions and release qualification remain open.

## Historical Norton Bass covariate forcing

- [x] Preserve additive per-generation coefficients and shared saved control history.
- [x] Verify analytic generation levels/rates, forecast replay and conditional fit.
- [x] Qualify Python, Julia and R and retain forcing limitations.

Source: d8d14ea5. Twenty-six unique focused Rust tests,
42 installed Python tests, 187 Julia assertions, R conformance and 163-page
documentation build passed. Successive-generation forcing, events, automatic
fit starts, uncertainty and scientific/release qualification remain open.

## Historical Lotka Volterra covariate forcing

- [x] Preserve named additive growth/competition controls and saved history.
- [x] Qualify analytic and independent references, fitting and all three bindings.
- [x] Review and retain limitations and completion evidence.

Source: 072206f5. Full local qualification passed 131 Rust core/C ABI/Polars
tests, 43 installed Python tests, 189 Julia assertions and R conformance.
The interacting SciPy reference, perturbed-start fit and held-out forecast pass.
Competition line coverage is 92.25 percent and core/C ABI line coverage is
85.37 percent. Docs build 163 pages. Local package metadata and exact source
inclusion pass. All 38 feature obligations remain required. This lane continues
in the same isolated worktree with existing source ownership.

## Historical multi-product covariate forcing

- [x] Support corrected named coefficients and explicit historical indexing.
- [x] Qualify analytic rates, independent trajectories, fits and Python/Julia/R.
- [x] Review and retain exact local evidence and outstanding obligations.

Source: 72c63338. Full local qualification passed 134 Rust core/C ABI/Polars
tests, 44 installed Python tests, 195 Julia assertions and R conformance.
Both coefficient layouts reproduce independent SciPy trajectories and forecasts.
The fit regression retained its original held-out tolerance and prompted the
shared relative objective-improvement correction. Competition line coverage is
92.47 percent, numerical-module coverage is 96.10 percent and core/C ABI line
coverage is 85.38 percent. Docs build 163 pages, and local package checks pass.
All 38 feature obligations remain required. Same isolated writing lane continues.

## Complementary goods covariate extension

- [x] Extend the shared forcing contract to intrinsic growth and complementarity.
- [x] Verify analytic/independent references, conditional fitting and bindings.
- [x] Retain review and scientific/release limitations.

Source: 95f3aeb3. Local qualification passed 24 affected Rust and bridge tests,
45 installed Python tests, 197 Julia assertions and R conformance. The analytic
and independent interacting references, perturbed conditional fit and held-out
forecast pass. Competition line coverage is 92.47 percent from focused tests.
The unchanged optimizer retains prior full-suite evidence. Docs build 163 pages
and local package checks pass. Lock-in forcing, event splits, uncertainty and
scientific/release qualification remain required. Same isolated lane continues.

## Lock in covariate and capacity forcing

- [x] Extend growth, feedback, competition and capacity through shared history.
- [x] Preserve clipping and qualify dynamic-capacity fits and diagnostics.
- [x] Verify independent references, Python/Julia/R and retained limitations.

Source: a5701060. Local qualification passed 27 affected Rust and bridge tests,
47 installed Python tests, 201 Julia assertions and R conformance. Independent
trajectories/rates, perturbed fitting and a held-out forecast pass. Capacity
decline/recovery and forced fits with a small base m have regression coverage.
Competition line coverage is 93.31 percent from focused tests. Docs build 163
pages and local package checks pass. Rates retain the clipped RHS convention,
and structural confounding remains explicit. Events, uncertainty, automatic
starts and scientific/release qualification remain required. Same lane continues.

## Fixed parameter events for ecosystem ODEs

- [x] Add validated persistent parameter overrides with continuous integrated state.
- [x] Verify exact boundaries, covariate composition and capacity projection.
- [x] Qualify independent references and Python/Julia/R, retaining inference gaps.

Source: 59a95e1b. Full local qualification passed 143 Rust core/C ABI/Polars
tests, 52 installed Python tests, 211 Julia assertions and R conformance.
Independent off-grid event references pass for all four model families.
Discovery advertises this contract. R event times retain scalar serialization.
Competition line coverage is 93.91 percent and core/C ABI coverage is 85.60
percent. Docs build 163 pages and local package checks pass. Event-time/override
inference, other families, uncertainty and scientific/release qualification
remain required. All 38 obligations remain open. Same isolated lane continues.

## Shared fixed events and historical Norton Bass

- [x] Reuse fixed-event validation and segmented integration for substitution.
- [x] Verify analytic generations, controls, fitting and saved-state forecasts.
- [x] Validate Python/Julia/R and retain successive-generation/inference gaps.

Source: 8ffe2a2e. Full local qualification passed 146 Rust core/C ABI/Polars
tests, 53 installed Python tests, 214 Julia assertions and R conformance.
Analytic generations, independent event/control trajectories, perturbed fitting
and a held-out forecast pass. Shared event line coverage is 99.24 percent,
substitution coverage is 90.50 percent and competition coverage is 93.93 percent.
Core/C ABI coverage is 85.76 percent. Docs build 163 pages and local package
checks pass. Successive-generation forcing/events, event inference, uncertainty
and scientific/release qualification remain required. Same lane continues.

## Successive generation forcing and events

- [x] Integrate launched Bass fractions with controls and fixed interventions.
- [x] Include time-varying capacity derivatives and explicit output-jump semantics.
- [x] Verify independent trajectories, fits, forecasts and Python/Julia/R.

Source: 418a600b. Full local qualification passed 152 Rust core/C ABI/Polars
tests, 54 installed Python tests, 217 Julia assertions and R conformance.
Independent fraction-space reference, perturbed fit and held-out forecast pass.
Finite-scale, first-launch history and event-boundary regressions are retained.
Substitution line coverage is 91.33 percent, mechanistic coverage is 90.68
percent and core/C ABI line coverage is 85.89 percent. Docs build 163 pages
and local package checks pass. Launch/event inference, threshold helpers,
uncertainty and scientific/release qualification remain required. Same lane
continues.

## Norton Bass baseline thresholds

- [x] Restore the historical baseline formulas with the correct named rates.
- [x] Verify summary transport and the substitution regression suite.

Twenty focused Rust tests and core Clippy passed. Baseline offsets retain the
historical inflection label and mark degenerate rates explicitly. Forced
thresholds and cross-binding qualification of this new field remain open.

- [x] Qualify the threshold field in installed Python, Julia and R.

The full native binding suites passed with 55 Python tests, 220 Julia assertions
and R conformance. The summary test checks the launch reference time, inflection
offset and saturation through each binding. Forced-trajectory analysis remains open.

## Norton Bass trajectory rate peaks

- [x] Report per-generation sampled net adoption rate peaks under forcing, with the analysis window and event-side convention explicit. (c38b6ad7)

All 21 native multi-generation tests and focused Clippy passed. The event
fixture verifies the right-hand rate at the event and excludes generations
whose launches follow the reporting window. Continuous-time peak location,
takeoff and saturation under forcing, and installed binding checks for the
trajectory field remain open.

- [x] Validate the forced trajectory peak through the installed language APIs.
  The fixed event raises the adoption rate to 50 exp(-0.2) at time 2.
  Python 3.15.2 passed 56 native tests, Julia 1.13.3 passed 223 assertions
  using the checked-in manifest and direct test entry point, and the installed
  R package passed native conformance. Julia's Pkg.test sandbox could resolve
  only JSON through 1.12.2 against the required 1.13.4, including with
  re-resolution disabled. The package-runner gate remains open.
  [Logs](evidence/norton-forced-peak-20261107).

- [x] Requalify the Julia package test runner with the checked-in manifest.
  Julia 1.13.3 passed `Pkg.instantiate(); Pkg.test()` in a fresh repo-local
  depot with General installed from the package server. The sandbox selected
  JSON 1.13.4 and passed all 223 assertions. The earlier failure used a stale
  General snapshot. [Package runner receipt](evidence/norton-forced-peak-20261107/julia-clean-pkg-test.log).

## Forced successive-generation adoption milestones

- [x] Locate the 10%, 50% and 90% intrinsic adoption crossings under forcing. (60cf3625)

The diagnostic integrates from launch, locates crossings between output
samples, and returns null for milestones beyond the requested horizon. It
uses the intrinsic launched fraction, so capacity jumps and cannibalization
of net levels do not define crossings. The independent piecewise exponential
fixture checks all three milestones before and between requested samples.
All 22 multi-generation tests and focused Clippy passed. Installed binding
checks for the milestone field and historical ODE threshold semantics remain
open.

- [x] Validate the milestone field in installed Python, Julia and R.
  Each consumer checks the analytic median after an event and verifies null
  saturation at the reporting horizon. The rebuilt wheel passed 56 native
  Python tests on 3.15.2, Julia 1.13.3 passed 225 assertions in Pkg.test,
  and installed R native conformance passed with the rebuilt Rust library.
  [Logs](evidence/norton-milestones-20261107).

## Historical cannibalization level milestones

- [x] Report sampled attainment of fixed baseline market fractions for historical net generation levels. (d8f454f4)

The historical ODE reports the first requested sample at or above each
fraction of the baseline market. Cannibalization can reverse net levels, so
the reported times express sampled attainment within the requested window.
A capacity event fixture confirms that the reference stays fixed. All 23
multi-generation tests and focused Clippy passed. Continuous-time historical
crossings and installed binding checks for this field remain open.

- [x] Validate sampled historical level milestones in installed Python, Julia and R.
  All three consumers preserved the 100-unit baseline through the 1,000-unit
  capacity event and returned the expected sample times. The rebuilt wheel
  passed 57 Python native tests, Julia Pkg.test passed 227 assertions, and
  the installed R package passed native conformance.
  [Logs](evidence/norton-historical-milestones-20261107).

## Scheduled heterogeneous Kairos hazards

- [x] Allow schedule boundaries to replace agent-specific innovation and imitation multipliers independently. (6f84a75b)

The native fixture moves innovation eligibility to a second agent at a
schedule boundary while preserving the first agent's completion time.
Malformed vector lengths are rejected. All 15 Rust network tests and focused
Clippy passed. Installed binding checks and continuous covariate hazard
integration remain open.

- [x] Qualify scheduled heterogeneous hazards in installed Python, Julia and R.
  The fixture verifies the two request identities and ticks and the first
  uninterrupted completion. The rebuilt wheel passed 58 Python native tests,
  Julia Pkg.test passed 229 assertions, and installed R conformance passed.
  [Logs](evidence/kairos-agent-hazards-20261107).

## Continuous Kairos hazard forcing

- [x] Integrate time-varying innovation and imitation hazards between deterministic scheduler events. (f128fbe8)

The opt-in linear hazard factors compose with scheduled rates and agent
multipliers. The native fixture checks zero-hazard wakeup, rounded request
and completion times, and seeded replay. All 16 network tests and focused
Clippy passed. Analytic clock qualification and installed binding checks
remain open.

- [x] Verify the continuous clock against an analytic linear hazard ramp
  and exercise zero-hazard wakeup in installed Python, Julia and R.
  The seeded unit exponential budget and cumulative hazard p*t²/2 locate
  the crossing at 0.435 model units, dispatched at tick 44. All 17 Rust
  network tests and focused Clippy passed. The wheel passed 59 Python
  tests, Julia Pkg.test passed 231 assertions, and installed R native
  conformance passed. [Logs](evidence/kairos-continuous-hazards-20261107).

- [x] Validate effective hazards without evaluating the unused unforced sum. (286bf0be)
  The regression uses a finite attenuated hazard whose unforced population
  sum would overflow. All 18 Rust network tests and focused Clippy passed.

- [x] Requalify the effective-hazard fix in installed Python, Julia and R.
  All three bindings ran the same overflow regression fixture and returned
  adoption counts of [0, 2]. The rebuilt wheel passed 59 native tests, Julia
  Pkg.test passed 231 assertions, and installed R native conformance passed.
  [Logs and fixture](evidence/kairos-effective-hazard-20261107).

- [x] Normalize heterogeneous hazard bases before aggregation. (2b89ea03)
  Two 1e308 agent weights now compose with an attenuating coefficient
  without overflowing the intermediate basis sum. All 19 Rust network
  tests and focused Clippy passed. Binding evidence for this edge case
  remains open.

- [x] Requalify heterogeneous hazard scaling in installed Python, Julia and R.
  The same large-weight fixture returned adoption counts of [0, 2] in all
  three bindings. The rebuilt wheel passed 59 Python native tests, Julia
  Pkg.test passed 231 assertions, and installed R conformance passed.
  [Logs and fixture](evidence/kairos-scaled-hazard-20261107).

## Automatic starts for forced Norton Bass fitting

- [x] Fit declared covariate coefficients without requiring user-supplied starts. (1220d258)

The existing nine base starts are extended with zero beta slopes and used
when a forced fit has no explicit start. The regression fits a historical
trajectory without start parameters and reaches RSS below 1e-8. All 23
multi-generation tests and focused Clippy passed. Installed consumer checks
and broader independent qualification of automatic forced fits remain open.

- [x] Qualify explicit and automatic starts against independent two-generation
  historical and successive-generation reference trajectories in all three
  installed bindings. Both cases retain fixed events and meet the existing
  held-out forecast tolerance. The automatic successive fit reached a feasible
  domain boundary during differentiation. The shared solver now uses the
  available one-sided difference when one probe fails, while preserving
  central differences in the interior. Full core tests and Clippy passed.
  [Validation logs](evidence/norton-automatic-starts-20261107).

## Automatic starts for forced competition models

- [x] Extend native competition fit starts to declared covariate slopes. (a5d2926d)

The shared competition solver uses its existing base starts with zero
covariate slopes when no explicit start is supplied. The independent
interacting Lotka–Volterra case converges below RSS 1e-10. All 14 competition
tests and focused Clippy passed. Independent automatic-start validation for
the other competition models and installed binding checks remain open.

- [x] Verify automatic complementary-goods fitting against the independent
  forced trajectory and held-out forecast. The explicit-start case continues
  to run with the same oracles. All four complementary tests and Clippy passed.
- [x] Improve automatic multi-product and lock-in starts for the independent
  forced fixtures. The covariate-aware start described below meets the
  existing convergence and held-out accuracy gates at 200 iterations
  through Rust, Python, Julia and R.

- [x] Qualify complementary-goods automatic starts in installed Python,
  Julia and R. Both start modes converge and reproduce the independent
  held-out forecast to 1e-5. The rebuilt wheel passed 62 native tests, Julia
  Pkg.test passed 239 assertions, and installed R conformance passed.
  [Validation logs](evidence/complementary-automatic-starts-20261107).

- [x] Qualify automatic interacting Lotka–Volterra fits in installed
  Python, Julia and R. Both start modes converge and reproduce the
  independent held-out forecast within 1e-5. The focused runs reuse
  the installed builds from the complementary-goods checkpoint.
  [Validation logs](evidence/lotka-automatic-starts-20261107).

- [x] Isolate automatic-start termination from held-out forecast accuracy.
  At 500 iterations both fixtures still report the iteration limit. Lock-in
  reaches the existing 1e-3 forecast tolerance (maximum error 5.06e-4),
  while multi-product misses it (maximum error 0.1383) despite low training
  RSS. These results were obtained through the installed Python native API
  with the checked-in independent fixtures and no explicit starting values.
  The convergence and forecast gates remain open. Keep the current defaults
  and tolerances while improving initialization or optimizer behavior.
  [Reproduction and results](evidence/competition-start-diagnostics-20261107).

- [x] Expose individual multistart outcomes and the selected zero-based start
  index in solver diagnostics. The competition fixture verifies that the
  selected loss is the lowest reported loss. All 14 competition tests and
  focused Clippy passed. Automatic multi-product and lock-in convergence
  remain open.

- [x] Verify per-start diagnostics in the three installed bindings and
  compare the unresolved 200-iteration automatic fits. The interacting
  Lotka–Volterra case reports three converged starts in Python, Julia and R,
  each selecting the minimum reported loss. All three starts hit the
  iteration limit in both multi-product and lock-in. Multi-product
  selects index 2 and lock-in selects index 1. These focused checks use
  rebuilt Python and Rust FFI artifacts.
  [Per-start results](evidence/competition-start-reports-20261107).

## Automatic starts for mechanistic Bass and Gompertz

- [x] Add observation-scaled starting capacities and a deterministic rate
  grid for mechanistic fits without explicit initial parameters. Declared
  slopes start at zero. Unused Gompertz shape parameters default to one
  and retain explicit caller values when provided. Both starting modes
  pass the existing held-out forecast checks for forced Bass and fixed-event
  Gompertz. All nine mechanistic tests and focused Clippy passed.
- [x] Qualify the automatic path in installed Python, Julia and R.
  The focused checks use a rebuilt wheel and FFI library. All three bindings
  converged without explicit starts and reproduced the held-out forced Bass
  and fixed-event Gompertz reference values within 0.001. The Gompertz checks
  also verify the default unused shape parameters.
  [Validation logs](evidence/mechanistic-automatic-starts-20261107).

- [x] Update the migration guide with the verified automatic mechanistic and
  Lotka–Volterra fit contracts and the per-start diagnostic fields.
  The documentation build generated 163 pages, validated internal links,
  and included the new diagnostic field in the rendered migration page.

## Automatic historical Logistic fit starts

- [x] Seed historical Logistic fits from observed capacity, midpoint and a
  deterministic rate grid for each event segment. Covariate slopes start
  at zero. Explicit starts use their existing path.
- [x] Exercise both start modes against the event forecast and independent
  additive covariate fixtures. Five Rust tests and focused Clippy passed.
- [x] Qualify automatic starts in installed Python, Julia and R.
  Focused checks against the rebuilt wheel and FFI verified convergence,
  event forecast accuracy within 0.001, and recovery of the additive
  capacity slope within 0.001 in all three languages.
  [Fixtures and results](evidence/logistic-automatic-starts-20261107).

## Historical Logistic conditional adoption rates

- [x] Expose analytic time derivatives holding pointwise controls fixed.
  The event sample uses the post-event curve and excludes jump impulses.
  Unrepresentable derivatives serialize as null, preserving finite levels.
  Five focused Rust tests and Clippy passed.
- [x] Qualify the installed bindings and Polars rate column. Focused checks
  against rebuilt native artifacts give level 100 and rate 50 at the event
  boundary. Python also checks the fixed-control derivative.
  [Fixture and logs](evidence/logistic-conditional-rates-20261107).
- [x] Extend the native JSON derivative contract to explicit time-varying
  control slopes. The chain rule includes capacity, growth and midpoint
  effects in the active event branch. Six Rust tests and focused Clippy
  passed.
- [x] Qualify total rates in installed bindings and expose control slopes
  through the Polars frame path.
  The Polars API maps declared controls to explicit slope columns.
  Focused rebuilt Python, Julia and R checks all returned rate 10 for
  the same analytic case. Both Python Logistic Polars tests passed.

- [x] Validate the Polars slope mapping against declared control names
  before reading columns. The installed Python regression rejects an
  unexpected control name and retains the analytic rate result. All four
  native tabular tests and both focused Python Logistic tests passed.

## Native integration checkpoint

- [x] Re-run local core, tabular, FFI and installed consumer suites after
  the historical fit and rate changes. Rust passed 162 tests, Python
  passed 65 tests, Julia Pkg.test passed 241 assertions, and installed R
  conformance passed.
  Rustfmt and Ruff passed. Basedpyright reported zero errors with the
  repo-local interpreter and one missing native-module-source warning.
  Migration policy reports 38 release obligations still open.
  [Local integration results](evidence/native-integration-checkpoint-20261107).

## Weighted mechanistic diffusion fitting

- [x] Apply normalized observation weights to mechanistic Bass and Gompertz
  residuals and their local Jacobian. Report weighted loss separately from
  descriptive metrics. The independent forced Bass fixture fits despite a
  zero-weight outlier of 1e100. Fourteen Rust tests and focused Clippy passed.
- [x] Qualify weighted mechanistic Bass fits in installed Python, Julia and R.
  Focused checks against rebuilt native artifacts confirmed convergence,
  40 positive observations and normalized weighted RSS below 1e-5 in all
  three bindings. Python used the Polars fit entry point.
  [Fixture and results](evidence/mechanistic-weights-20261107).

- [x] Qualify weighted fixed-event Gompertz in installed Python, Julia and R.
  With an interior zero-weight outlier of 1e100, all three fits converged,
  reported 16 positive observations, and reproduced the analytic held-out
  level within 0.001. Python used the Polars fit and forecast entry points.
  These focused checks reuse the native artifacts from the Bass weight checkpoint.
  [Fixture and results](evidence/gompertz-weights-20261107).

## Weighted historical Logistic fitting

- [x] Apply normalized weights to the objective and local rank. Automatic
  starts use positive-weight observations for capacity and midpoint
  initialization. The analytic event fixture excludes an interior outlier
  of 1e100 and preserves the held-out forecast within 0.001. All eleven
  focused Rust tests and Clippy passed.
- [x] Qualify installed Python, Julia and R weighted historical Logistic fits.
  Focused rebuilt native checks report convergence, 29 positive rows and
  null AIC. Each held-out forecast meets the 0.001 analytic tolerance.
  Python uses the Polars fit and forecast entry points.
  [Fixture and results](evidence/historical-logistic-weights-20261107).

## Weighted competition fitting

- [x] Apply one weight per time row to all series in the shared competition
  solver. Positive rows set the outcome scale and automatic capacity
  starts. A row that supplies the conditioned initial state is excluded
  from the weighted residual objective. Weighted local rank and loss are
  separate from descriptive metrics. The independent interacting
  Lotka–Volterra fixture excludes a zero-weight outlier in both outputs.
  Eighteen focused Rust tests and Clippy passed.
- [x] Qualify weighted competition fits in installed Python, Julia and R.

- [x] Qualify weighted Lotka–Volterra in installed Python, Julia and R.
  All three bindings excluded the same two-output outlier and returned
  normalized weighted RSS below 1e-10. Python used the Polars fit API.
  Other competition models still require model-specific binding evidence.
  [Fixture and results](evidence/lotka-competition-weights-20261107).

- [x] Verify weighted complementary goods, lock-in and multi-product fits
  and held-out forecasts in all three bindings. The focused fixtures use
  explicit reference starts and replace both observations in one zero-weight
  row with 1e100. Every fit converged and the largest held-out absolute
  error was below 1.5e-9. Python used Polars for fitting. This evidence
  qualifies row weights with explicit starts. Automatic lock-in and
  multi-product convergence remain separate open tasks.
  [Fixtures and results](evidence/remaining-competition-weights-20261107).

## Weighted Norton Bass fitting

- [x] Apply one weight per time row across all generations, including
  covariate and event fits. Positive rows determine initialization and
  outcome scale. The independent historical fixture excludes an outlier
  in both outputs and preserves the held-out forecast within 1e-3.
  All 24 multi-generation Rust tests and focused Clippy passed.
- [x] Qualify weights for historical and successive variants in installed
  Python, Julia and R.
  Focused rebuilt binding checks used explicit starts and excluded a
  zero-weight outlier in both generations. Each fit converged with
  normalized weighted RSS below 1e-8 and held-out error below 1e-3.
  Python used the Polars fit entry point.
  [Fixtures and results](evidence/norton-weights-20261107).

- [x] Synchronize the weighted-fit migration guide with the verified
  model coverage, row-wise weighting and initial-state conditioning.
  The site built 163 pages with all internal links valid, and the
  rendered guide contains the updated multi-output contract.

## Release ledger reconciliation

- [x] Update the ledger with bounded hierarchical/mixture and latent/regime
  evidence. Refresh the remaining work for forcing, substitution, competition
  and Kairos based on the native checkpoints. Development policy and four
  mutation tests passed. The release check still rejects all 38 open obligations.

- [x] Validate ledger status values and require existing repository-local
  evidence files for bounded and qualified entries. All six policy
  regression cases and the live ledger check passed. The 38 release
  obligations remain open.

## Native batch fitting and forecasting

- [x] Restore independent batch fitting and prediction at the Rust kernel
  boundary. Preserve input order and each fit state or structured error.
  Forecast grids may have different lengths. All eight native boundary
  tests passed, including a failed middle fit and two distinct forecasts.
- [x] Expose native JSON batch fitting and forecasting through Python, Julia
  and R. Focused rebuilt checks preserve the failed middle fit and return
  forecasts on independent grids.
- [x] Expose batch operations through the Polars frame API. Python
  `fit_frames` and `predict_frames` orchestrate the existing native frame
  entry points in input order. The installed regression covers lazy frames,
  the failed middle fit, independent forecast lengths and identity columns.

- [x] Verify the batch JSON boundary rejects malformed envelopes and
  isolates an invalid dataset between two successful predictions. All nine
  native boundary tests passed. Record batch fitting as bounded native in
  the release ledger, with broader model and workload qualification open.

- [x] Allow opt-in parallel native JSON batches via `payload.workers`
  (1–32, default 1). Contiguous work chunks rejoin in original order.
  The regression compares complete serial and parallel envelopes,
  including a failed middle dataset, and rejects an invalid worker count.
  All nine native boundary tests and focused Clippy passed.
- [x] Qualify parallel workloads in the installed bindings and measure
  throughput across representative model families.
  Rebuilt Python, Julia and R consumers returned identical ordered serial
  and two-worker fit results. One local debug-build timing measured 48
  Logistic requests at 0.099/0.063 seconds and 16 forced Bass requests at
  3.944/2.317 seconds. These are local measurements, with broader workload
  and release qualification open. [Logs](evidence/parallel-batch-20261107).

- [x] Handle worker startup failures without losing batch ordering. A
  chunk whose worker could not start runs inline. A worker panic produces
  one structured error per affected dataset, and other chunks retain
  their results. All nine native boundary tests and focused Clippy passed.
  The operating-system failure paths were reviewed but not fault-injected.

## Native adopter classification

- [x] Restore the historical five adopter categories in Rust Polars from
  sample-weighted adoption time moments. Keep inclusive boundaries and
  original row identity. Reject negative rates and zero total weight.
  The fixture covers all five categories and weights whose raw sum would
  overflow. All five tabular tests and Clippy passed.
- [x] Expose classification through Python, Julia and R. All installed
  bindings returned the five historical labels on the extreme-weight fixture.
- [x] Connect the
  fitted model adoption-rate path where scientifically valid.

- [x] Connect Python fitted single-series models to native adopter
  classification and expose the plain Logistic analytic derivative.
  The installed Polars regression verifies the midpoint rate and
  category endpoints while retaining identity columns.
- [x] Qualify model-derived rates for other plain diffusion curves and
  connect fitted-state classification in Julia and R.

- [x] Connect fitted-state classification in installed Julia and R.
  Both consumers fit a Logistic curve, predict native rates, and
  reproduce the analytic midpoint rate of 25 and endpoint categories.
  Both reject prediction outputs with multiple rate series.

- [x] Add analytic rates to plain Bass, Gompertz and Fisher–Pry predictions.
  The native boundary regression verifies reference rates at the Bass
  origin, Gompertz origin and Fisher–Pry midpoint. All ten boundary tests
  and focused Clippy passed. Installed checks of these three rate paths
  and extreme-domain rate qualification remain open.

- [x] Verify plain Bass, Gompertz and Fisher–Pry analytic rates across rebuilt
  Python, Julia and R bindings. Python uses the Polars rate column. All nine
  model-binding comparisons meet an absolute tolerance of 1e-10 at the
  reference points. Extreme-domain coverage remains open.
  [Fixture and results](evidence/plain-diffusion-rates-20261107).

- [x] Stabilize plain Bass levels and rates when q/p exceeds the
  representable range. The native regression checks the origin and
  transition midpoint for p=1e-200 and q=1e200. Eleven boundary tests
  and focused Clippy passed. Installed edge-case checks remain open.

- [x] Verify extreme plain Bass predictions in installed Python, Julia and R.
  With p=1e-200 and q=1e200, all three return midpoint level 0.5 and
  rate q/4 within 1e-11 (absolute for level, relative for rate).
  Python and Julia reproduce the origin rate exactly. R's JSON round trip
  differs by 4.44e-16 relative and passes the 1e-11 rate tolerance.

- [x] Route Python adopter frames through Arrow IPC and the Rust Polars
  classifier. The rebuilt wheel passes the fitted Logistic regression,
  including the rate and preserved identity column.

- [x] Exercise fitted Bass, Gompertz and Fisher–Pry classification through
  the installed Python Polars API. All three regressions return aligned
  positive rate series and multiple categories. The ledger now records
  bounded native support. Broader rate-domain and grid qualification remains open.

## Polars time-series preprocessing

- [x] Restore cumulative sums and full-window trailing averages through
  native Polars series operations in Python. The installed test verifies
  sample order, warmup nulls and rejection of a zero window.
- [x] Restore timestamp aggregation, decomposition and seasonal model
  preprocessing, and qualify Julia and R consumer paths.

- [x] Restore historical weekly sums in Polars with fixed 168-hour
  left-closed buckets anchored at the earliest timestamp. The installed
  regression uses unsorted dates, an exact boundary and an empty week.
  Daily and monthly aggregation and seasonal preprocessing remain open.

- [x] Restore daily and month-end Polars aggregation. Calendar periods
  between the first and last observations are filled with zero sums,
  while weekly aggregation retains its first-observation anchor and
  omits empty weeks. Both installed aggregation tests passed, including
  the leap-year month-end label. Seasonal preprocessing remains open.

- [x] Expose native cumulative sums and full-window trailing averages for
  finite numeric arrays in Julia and R. Both installed consumers match
  the Python Polars example, including two null warmup positions.
  Focused Rust Clippy passed. Calendar and seasonal consumer paths remain open.

- [x] Use compensated accumulation for native preprocessing. The
  rolling-window regression recovers the average of [1, 3] after 1e100
  leaves the window. All twelve native boundary tests passed.

- [x] Verify the compensated rolling average through the rebuilt Rust FFI
  in Julia and R and retain the case in both native test suites. Both
  report a null warmup value, then 5e99 and 2.

- [x] Expose calendar-date aggregation in Julia and R using the Rust
  kernel and the existing locked Chrono version. Daily and monthly
  buckets fill gaps, while weekly buckets are anchored at the earliest
  date and omit empty periods. Installed Julia and R checks match the
  leap-year Polars fixture. Thirteen Rust boundary tests and Clippy passed.
  Timestamp and offset handling are verified in the checks below.

- [x] Implement a native local-linear tricube smoother on evenly spaced
  samples as a building block for seasonal decomposition. The boundary
  fixture reproduces a linear series at its edges and interior. Fourteen
  native boundary tests and focused Clippy passed. STL seasonal subseries,
  low-pass filtering and robust iterations remain open.

- [x] Implement the seasonal decomposition inner loop with phase
  subseries smoothing, endpoint extension and centered low-pass filtering.
  An analytic linear-plus-periodic fixture recovers all three components
  within 1e-10. Fifteen Rust boundary tests and focused Clippy passed.
  Robust reweighting, historical STL option parity and installed binding
  qualification remain open.

- [x] Expose the bounded native seasonal decomposition in Python, Julia
  and R. All rebuilt consumers recover the analytic trend, seasonal and
  residual components within 1e-10. Python returns a Polars frame.
  Robust reweighting and full historical STL option parity remain open.

- [x] Add opt-in robust seasonal reweighting at the Rust JSON boundary.
  Bisquare residual weights affect both phase and trend smoothers.
  The analytic outlier fixture keeps a 100-unit spike in the residual
  within 0.1. All 16 native boundary tests and focused Clippy passed.
- [x] Expose and verify robust mode in Python, Julia and R. Each rebuilt
  binding retained the 100-unit spike within 0.1 in the residual.
- [x] Qualify historical STL against independent R fixtures for the
  tested configurations. See the robust, mixed smoother and partial
  cycle checks below.

- [x] Measure component agreement with independent R stats::stl on a
  deterministic perturbed periodic series with a 100-unit spike. Reference
  configuration uses period 4, spans 7/13/5, local degrees 1, jumps 1 and
  five inner iterations. Robust mode uses five outer iterations.
  Nonrobust maximum component discrepancy is 0.001193. Robust discrepancy
  is much larger (100.0763 for residuals), so the existing analytic outlier
  test cannot establish historical STL parity. Endpoint extrapolation,
  weighting behavior and option parity remain open.
  [Reference inputs, outputs and differences](evidence/stl-reference-20261107).

- [x] Evaluate nonrobust phase endpoints directly with local regression.
  The saved R STL reference now agrees within 1e-10 on all components.
  Applying the same endpoint convention in robust mode moved the analytic
  spike residual to 103, outside its existing tolerance, and did not close
  the independent robust discrepancy. Robust mode retains its prior
  endpoint convention pending parity work. Seventeen Rust boundary tests
  and focused Clippy passed.

- [x] Reproduce the independent fixture with the checked-in R checker and
  compare it against the rebuilt Rust library through installed innovate.R.
  The largest nonrobust component difference was 1.57e-13. Robust maximum
  differences were 30.24 for seasonal, 69.83 for trend and 100.08 for
  residual, leaving robust parity open.

- [x] Isolate the robust reference's dependence on outer iterations.
  Across iterations 1–6, the spike weight alternates between zero and one
  and 4–25 observations have zero weight. At five iterations R assigns
  zero weight to the spike but reports a zero spike residual. Investigate
  local fits with no positive weights before changing the Rust endpoint
  convention or adopting this fixture as a robust outlier-recovery oracle.

- [x] Trace the robust fallback in the R 4.3 STL source. When a local
  window has no positive weight, R uses the input at that index for an
  in-series fit and the adjacent smoothed value for a phase extension.
  Rust instead finds the nearest positive-weight observation. R also
  suppresses slope estimation when the weighted coordinate standard
  deviation is at most 0.001 times the series coordinate range. An
  isolated Rust change to reject numerically degenerate slopes failed
  the existing analytic spike-recovery test and was reverted. Qualify
  the fallback and slope rules together against the independent fixture
  and retain the analytic spike test when evaluating the resulting
  behavior.
  [R STL source](https://github.com/wch/r-source/blob/R-4-3-branch/src/library/stats/src/stl.f).

- [x] Test the combined R-compatible robust rules through installed R.
  Direct phase extension, observed-value fallback inside the series,
  adjacent smooth fallback outside it, the weighted spread threshold,
  and the R bisquare cutoffs reduced maximum robust discrepancy from
  100.08 to 1.41e-10 on the saved fixture. The combined prototype
  failed the existing analytic spike-recovery test when applied to the
  default. Restored the default. Add explicit historical compatibility
  semantics so both the independent parity fixture and the analytic
  spike-recovery test can be enforced.

- [x] Implement explicit historical STL fit semantics in the Rust
  preprocessing request (`historical: true`). The robust independent
  fixture passes within 1e-9, and the default analytic spike-recovery
  test still passes. All 18 native boundary tests passed.
- [x] Expose historical STL mode in Python, Julia and R. The installed
  Python wheel (CPython 3.15.2), Julia project (1.13.3), and installed R
  package (4.3.0) each match the robust reference within 1e-9.
- [x] Qualify configurable historical STL iterations, spans, degrees
  and jumps on the reference configurations below.

- [x] Add bounded inner and outer STL iteration controls to the native
  preprocessing request. The installed R generic bridge agreed with
  stats::stl within 1.56e-11 across nine combinations of inner counts
  1/2/5 and outer counts 0/1/3. All 18 native boundary tests and focused
  Clippy passed.
- [x] Expose iteration controls in Python, Julia and R helpers. At two
  inner and three outer passes the rebuilt Python wheel and Julia
  project each agreed with their generic native entry point and
  differed from the defaults. Installed R agreed with stats::stl
  within 1.56e-11.
- [x] Qualify historical span, degree and jump options. The pre-migration
  Python STL wrapper forwarded constructor keyword options.

- [x] Add seasonal and low-pass spans to the native STL request.
  Historical mode allows spans longer than the local subseries and
  expands the regression radius accordingly. With trend/low-pass
  spans 17/9, seasonal spans 9 and 25 matched stats::stl within 1.24e-12
  through installed R. All 18 native boundary tests and focused Clippy
  passed.
- [x] Expose optional seasonal and low-pass spans in the Python, Julia
  and R helpers. The rebuilt Python wheel, Julia project and installed
  R package each matched stats::stl within 1e-9 with seasonal/trend/
  low-pass spans 25/17/9 and inner/outer counts 2/3. Calls omitting both
  new options also passed in all three runtimes.
- [x] Qualify historical degree and jump options.

- [x] Implement degree 0/1 and positive jump controls for each STL
  smoother in Rust. Four mixed degree/jump configurations matched
  stats::stl through installed R within 6.37e-12, including off-grid
  terminal points. All 18 native boundary tests and focused Clippy passed.
- [x] Expose degree and jump controls in the three binding helpers.
  The rebuilt Python wheel, Julia project, and installed R package
  matched stats::stl within 6.37e-12 at seasonal/trend/low-pass degrees
  0/1/0 and jumps 2/3/5 with robust reweighting.

- [x] Qualify historical STL phase alignment on series that end in a
  partial cycle. All eight length/period combinations matched the
  independent R reference within 2.10e-13, including 13 observations
  with a seasonal span exceeding each phase subseries. The R checker
  asserts agreement for each component and verifies output shape.

- [x] Derive the historical STL trend span when omitted and expose
  omission through the three binding signatures. The updated R helper
  with rebuilt FFI matched stats::stl within 1e-9 across nine period
  and seasonal span pairs when the low-pass span was held equal.
  Odd periods use the next strictly greater odd low-pass span in
  Innovate, while R defaults to the period itself.
- [x] Verify the inferred span through rebuilt Python and Julia. For
  period 5 with the default seasonal span, both runtimes agreed exactly
  with explicit trend span 11 and differed from explicit span 13.


- [x] Capture a reproducible independent stationary SARIMA reference
  with fixed nonseasonal and seasonal AR/MA coefficients and aligned
  one-step fitted values. The historical Python wrapper estimated a
  no-trend SARIMAX model and returned fitted values in input order.
  [Reference generator and fixture](evidence/sarima-reference).
- [x] Implement fixed-coefficient filtering, differenced orders and
  parameter estimation in Rust. The stationary, integrated and mixed
  reference checks below define the verified scope.

- [x] Add stationary fixed-coefficient SARIMA filtering to the Rust
  preprocessing request. Multiplicative AR/MA polynomials, stationary
  covariance and innovations yield aligned fitted values and residuals.
  Corrected the R oracle to use prefix forecasts because R's fitted
  residuals are standardized by the prediction variance. The independent
  fixture agrees within 1e-10 at all 40 observations. All 19 native
  boundary tests and focused Clippy passed.

- [x] Support ordinary and seasonal differencing in the fixed-coefficient
  Rust SARIMA request, reconstructing fitted values on the observation
  scale. Combined differencing passed an analytic trend/seasonal/shock
  fixture with nulls for the warmup rows. All 20 native boundary tests
  and focused Clippy passed. This conditional initialization still
  requires comparison with historical SARIMAX state-space initialization.

- [x] Verify the fixed-coefficient SARIMA request through the rebuilt
  Python wheel (3.15.2), Julia (1.13.3) and installed R (4.3.0) generic
  entry points. Each matched the stationary forecast fixture within
  1e-10 and retained four warmup positions for combined differencing.
  R now encodes singleton SARIMA coefficients as arrays and was verified
  with a one-coefficient AR model and a zero-coefficient integrated model.

- [x] Compute the concentrated Gaussian likelihood from SARIMA
  innovations and their prediction variances. The stationary fixture
  agrees with R logLik within 1e-10. Differenced requests use the
  likelihood conditional on the warmup observations. Degenerate
  likelihoods return null while fitted values remain available.
  All 20 native boundary tests and focused Clippy passed.

- [x] Add native SARIMA coefficient estimation using deterministic
  coordinate search and the concentrated likelihood. The AR(1) estimate
  matches R within 2e-6 and its log likelihood within 1e-9. All 21 native
  boundary tests and focused Clippy passed.
- [x] Qualify the mixed seasonal AR/MA fixture and the three higher-order
  optimizer cases below. Verify the Python, Julia and R fitting APIs.
- [ ] Extend optimizer qualification to difficult parameter regions
  beyond the recorded cases.

- [x] Compare a mixed SARIMA(1,0,1)(1,0,1)[4] likelihood fit on 240
  simulated observations with R maximum likelihood. The installed R
  bridge ran the Rust estimator and the reproducible checker verifies
  coefficient agreement within 5e-4 and log likelihood within 1e-6.
  Broader optimizer and differenced initialization qualification remain open.

- [x] Expose native SARIMA fitting through Python, Julia and R helpers.
  Installed runtime checks retained the observation scale and four
  warmup entries for combined differencing, represented as Polars
  nulls, Julia missing values and R numeric NAs. The Python wheel also
  exercised coefficient estimation through the helper on an AR(1).

- [x] Compare combined ordinary and seasonal differencing on an
  integrated AR(1) reference. The Rust fit through installed R agrees
  with stats::arima within 5e-6 in coefficient and 1e-6 in likelihood.
  Fixed-coefficient prefix predictions at rows 7–40 agree within 1e-8.
  Historical SARIMAX initialization and warmup parity remain open.

- [x] Capture the historical SARIMAX initialization reference at fixed
  coefficients for combined differencing, with and without an AR term.
  Statsmodels 0.14.6 returns finite fitted values in the four likelihood
  burn rows, where the native conditional path currently returns null.
  The reference uses the retained historical environment only to
  generate evidence.
- [x] Implement historical SARIMAX initialization and compare every
  fitted value against both fixed-coefficient fixtures. Verify the
  estimated integrated and mixed historical cases below.

- [x] Add opt-in historical SARIMA state-space initialization in Rust.
  The differencing states start with diffuse variance and the ARMA
  states with stationary covariance. Fitted values include the
  warmup rows, which are excluded from the concentrated likelihood.
  Both fixed-coefficient historical fixtures agree within 1e-8 at
  every observation. All 22 native boundary tests and focused Clippy
  passed.
- [x] Expose historical initialization in all three fitting helpers.
  On 245 observations with ordinary and seasonal differencing, the
  rebuilt Python wheel, Julia project and installed R package matched
  statsmodels fitted values within 1e-4. The native Python request
  differed by 5.42e-6 in the estimated AR coefficient and 3.97e-9 in
  log likelihood. Broader mixed-order historical qualification remains open.

- [x] Qualify one historical mixed SARIMA(1,1,1)(1,1,1)[4] estimated
  fit on 405 observations against statsmodels. The Python wheel
  matched coefficients within 6.86e-5, fitted values within 8.86e-5
  including warmup, and likelihood within 1.80e-7. The checker separates
  reference generation from native validation. Optimizer behavior
  across higher orders and difficult parameter regions remains open.

- [x] Improve the SARIMA search with a joint step along the direction of
  each coordinate sweep and allow up to 2,048 sweeps. The higher-order
  checker covers three ARMA fits. ARMA(1,2) previously exhausted the
  search budget and now converges to a different basin with likelihood
  0.495 higher than the reference R fit. The other two likelihoods
  agree within 1e-8. All 22 boundary tests and focused Clippy passed.

- [x] Verify the revised optimizer in the rebuilt Python wheel and Julia
  against the installed R binding on the previously nonconvergent
  ARMA(1,2) series. The public fitting helpers returned all 200 fitted
  values and agreed within 1e-10 across runtimes.

- [x] Schedule Kairos waiting queue limits. Reducing a limit to zero
  preserves an existing waiter and blocks later arrivals while service
  is full. The result reports the effective limit alongside the initial
  limit. All 20 network tests and focused Clippy passed.
- [x] Verify scheduled queue limits through the rebuilt Python wheel,
  Julia project and installed R package. All three returned matching
  counts, requests, balks and completions. The later arrival balked
  at tick 4, the existing waiter completed, and the effective
  queue limit was zero.

- [x] Schedule Kairos wait deadlines for new waiters. Existing waiters
  retain their assigned deadlines. A regression shows a later arrival
  expiring at tick 5 under the new limit while the earlier waiter is
  admitted. All 21 network tests and focused Clippy passed.
- [x] Verify scheduled wait deadlines through the rebuilt Python wheel,
  Julia and installed R. Counts, requests, abandonments and completions
  matched exactly. The later waiter expired at tick 5 and the original
  waiter completed under its retained deadline.

- [x] Allow scheduled queue policies to clear wait and length restrictions
  with explicit null. Omitted fields retain their current values. The
  regression clears both limits and verifies all three agents complete
  without balking or abandonment. All 22 network tests and focused
  Clippy passed.
- [x] Verify explicit null queue changes through the rebuilt Python
  wheel, Julia and installed R. All three agree on adoption counts,
  request and completion traces, no balks or abandonments, and null
  effective wait and queue limits.

- [x] Extend the Rust calendar request to ISO timestamps. Offset-aware
  inputs use UTC for calendar boundaries and weekly labels. Naive inputs
  retain wall-clock times. Weekly buckets use elapsed 168-hour intervals
  from the earliest instant. Mixed offset and naive inputs are rejected.
  The exact weekly boundary fixture, 23 native tests and focused Clippy
  passed.
- [x] Qualify weekly timestamp aggregation in Julia and installed R.
  Both matched the exact boundary fixture with explicit offsets.
  Julia DateTime retains wall time, and R POSIXt is formatted in UTC
  with microsecond precision before dispatch.


- [x] Verify UTC day and month-end buckets through Julia and installed R.
  Brisbane March 1 at 00:30 falls in the February 29 UTC bucket. Both
  bindings returned daily sums 2/0/3 with the missing day filled and
  month-end sums 2/3 labelled February 29 and March 31. Julia used
  offset-bearing ISO strings and R used Brisbane POSIXct inputs.

## Native causal evaluation

- [x] Port the four historical descriptive summaries to the Rust
  `summarize_causal` request. The analytic boundary test covers
  pre/post differences, counterfactual percent change, missing
  pre-event summaries and subgroup intervals. Duplicate group labels
  are rejected. All 24 boundary tests and focused Clippy passed.
- [x] Verify all four summaries through the rebuilt Python wheel,
  Julia and installed R. The checks cover singleton arrays, a missing
  pre-event result, and the analytic subgroup interval. R preserves
  singleton arrays for this operation. Record bounded native status
  in the migration ledger.
- [ ] Qualify the broader historical and planned causal workflows.

- [x] Port historical covariate balance summaries with pooled sample
  variance and absolute standardized mean differences clipped at one.
  Shared covariates are compared, and the historical zero-variance
  score is retained. Compared groups require at least two observations.
  The analytic fixture verifies the pooled variance and threshold
  counts. All 24 Rust boundary tests and focused Clippy passed.
- [x] Verify covariate balance through the rebuilt Python wheel, Julia
  and installed R. All three matched the analytic pooled SMD within
  1e-12 and identified one of two covariates as balanced.

- [x] Port the historical point E-value and directional sensitivity
  scenario. The native request validates risk ratio, gamma and bias
  direction, and labels the scenario margin as a historical heuristic.
  Analytic reciprocal-risk and one-sided scenario checks passed with
  all 24 Rust boundary tests and focused Clippy.
- [x] Port paired and grid sensitivity and verify all four sensitivity
  requests in the rebuilt Python wheel, Julia and installed R. Each
  matched the analytic one-pair bound, singleton gamma-grid result,
  directional scenario and reciprocal-risk E-value.

- [x] Port the historical paired sensitivity bounds and gamma-grid
  summaries. Both identify their outputs as heuristic. Paired bounds
  reject misaligned arrays, and the grid preserves input order.
  Analytic tests verify the bounds and grid values. R marshals singleton
  arrays for these requests. All 24 boundary tests and focused Clippy
  passed. Installed sensitivity validation is recorded above.

- [x] Restore the historical parameter-override forecast comparison
  through the native predictor. The Bass fixture doubles market size
  and verifies a 100% increase after adoption begins. A zero baseline
  produces a null percentage difference. All 24 boundary tests and
  focused Clippy passed.
- [x] Verify the Bass forecast comparison in the rebuilt Python wheel,
  Julia and installed R. All three preserve the zero-baseline null
  and match the analytic 100% increase at the next observation.
  Python uses the new forecast_model payload key because model is
  reserved by execute. Julia and R exercised the retained model key.
- [ ] Qualify additional historical counterfactual workflows.

- [x] Support counterfactual replacement of named covariate trajectories
  on the baseline prediction grid. The Logistic fixture changes only
  price and moves the midpoint prediction from 50 to 60. All 24 Rust
  boundary tests and focused Clippy passed.
- [x] Verify covariate overrides through the rebuilt Python wheel, Julia
  and installed R. The same Logistic scenario returns 50 at baseline,
  60 after replacing price, and a difference of 10. The R check uses
  a scalar input for the one-element time and covariate arrays.

- [x] Compare counterfactual forecasts with multiple product trajectories.
  The response preserves column labels and returns aligned row matrices
  for levels, differences and percentage differences. The Lock-in
  fixture verifies both the changed product and a zero baseline product.
  All 24 native boundary tests and focused Clippy passed.
- [x] Verify the Lock-in counterfactual through the rebuilt Python wheel,
  Julia and installed R. Each retains both product columns, returns
  differences of 20 and 0, and reports 20% and null respectively.

- [x] Restore uncertainty metadata summaries. The native request
  preserves the supplied interval and estimand, reports its width and
  approximate standard error under a 95% normal interval, and records
  whether it excludes zero. The fixture checks a bound at zero and
  rejects reversed bounds. All 24 boundary tests and focused Clippy passed.
- [x] Verify interval summaries through the rebuilt Python wheel, Julia
  and installed R. All three report width 3.92 and approximate standard
  error 1, retain the bootstrap count, and include zero at the boundary.

- [x] Port historical causal evidence checks for supporting records,
  caveats, evidence level and sensitivity analysis. The result reports
  checklist status and issues. External release approval remains a
  separate decision. All 24 Rust boundary tests and focused Clippy passed.
- [x] Verify evidence checks through the rebuilt Python wheel, Julia
  and installed R. All three report one issue for the medium-evidence
  fixture without sensitivity analysis and pass when it is recorded.
  R also exercised singleton evidence and caveat arrays.
- [x] Restore causal model cards and assumption records with structured
  JSON and Markdown assumption output. The card preserves a supplied
  creation date and uses null when omitted for reproducible output.
  All 24 Rust boundary tests and focused Clippy passed.
- [x] Verify model cards and assumption records through the rebuilt
  Python wheel, Julia and installed R. All preserve the supplied date,
  the default version and the singleton assumption. Assumption Markdown
  retains the heading and omits the absent check description.

- [x] Restore the historical unadjusted treatment contrast and subgroup
  comparisons. The response identifies the difference-in-means method
  and reports counts and null estimates for subgroups missing either arm.
  All 24 Rust boundary tests and focused Clippy passed.
- [x] Restore treatment-contrast bootstrap intervals and verify installed
  consumers. Policy contracts and Arrow handoff are tracked separately below.

- [x] Add deterministic stratified percentile bootstrap intervals for
  unadjusted treatment contrasts. Resampling within each arm keeps the
  observed arm sizes and avoids empty arms. The historical pooled
  resampling scheme remains a separate parity obligation. All 24
  native boundary tests and focused Clippy passed.
- [x] Verify pooled and stratified intervals through the rebuilt Python
  wheel, Julia and installed R. Both schemes return the analytic bounds
  of 3. The three bindings agree exactly on draw accounting and results.

- [x] Restore pooled treatment/outcome bootstrap resampling as an
  explicit scheme. Empty-arm draws are counted and omitted from the
  percentiles. Fewer than two valid draws produce null interval bounds.
  The seeded fixture verifies replay, draw accounting and the analytic
  interval. All 24 native boundary tests and focused Clippy passed.

- [x] Restore intervention contracts with comparator, timing, rollout
  and spillover fields. The validator rejects reversed times and an
  outcome window ending before intervention, including at start zero.
  All 24 Rust boundary tests and focused Clippy passed.
- [x] Restore causal model contract checks. High unobserved confounding
  risk requires sensitivity analysis, and the historical case-insensitive
  name heuristic rejects post-treatment confounder names. All 24 native
  boundary tests and focused Clippy passed.
- [x] Qualify both policy contracts in the rebuilt Python wheel, Julia
  and installed R. Each retained the singleton spillover and confounder
  lists and rejected high risk without sensitivity analysis.

- [x] Restore the treatment contrast Arrow result schema through a
  Polars frame projection. The rebuilt Python wheel round-tripped
  Float64 estimates and interval bounds plus Int32 draw counts through
  Arrow IPC read by PyArrow 26. Absent intervals retain null bounds
  and zero draws.
- [x] Restore causal model metadata and evidence handoff, and qualify
  other Arrow consumers.

- [x] Restore policy metadata and causal evidence handoff using the
  native intervention and causal model validators. The rebuilt Python
  wheel preserved assumptions and estimand fields, and PyArrow read
  the Polars IPC metadata row with an Int32 confounder count.
  All 24 Rust boundary tests and focused Clippy passed.
- [x] Verify policy evidence through Julia and installed R. Both
  preserve intervention metadata, the confounder list and the
  identifying assumption. R also exercised singleton list encoding.
- [x] Verify the Polars policy metadata and treatment contrast IPC
  streams in Julia 1.13.3 with Arrow 2.8.1 in an isolated project.
  Use write_ipc_stream with uncompressed output and the oldest Polars
  compatibility level. The reader preserves Int32 counts, Float64
  estimates, metadata strings, and missing interval bounds.
- [x] Verify R Arrow IPC readback. R 4.3.0 with the available
  macOS ARM64 Arrow 20.0.0.2 binary read the Polars streams, the
  uncompressed oldest-compatibility Polars files, and the PyArrow files.
  All preserved metadata, Int32 counts, Float64 estimates and null bounds.
  The Arrow binary reports that it was built under R 4.3.3.
- [x] Trace the Arrow.jl 2.8.1 file read failure to its message iterator.
  After the eight-byte file header it expects a continuation marker
  and stops on the first Polars metadata word (0x00000004). The PyArrow
  file and the Polars stream start their messages with 0xffffffff.
  Document the qualified stream handoff in the Julia binding README.
- [ ] Direct Polars IPC file compatibility in Arrow.jl remains open.

- [x] Isolate the Julia IPC file issue to the tested writer path.
  PyArrow 26 rewrote each qualified Polars stream to an IPC file.
  Arrow.jl 2.8.1 read the files and matched the stream column names,
  element types, values and nulls. The equivalent Polars-written
  files yielded empty column sets. Use the qualified Polars stream
  path for direct Julia exchange.

- [x] Restore causal policy specification serialization. Both contracts
  are validated, observation counts derive from aligned data columns or
  survive a metadata-only round trip, and data rows stay out of the
  exported specification. The Rust fixture checks round trip and rejects
  a ragged dataset. All 24 boundary tests and focused Clippy passed.
- [x] Qualify policy specification round trips through installed bindings
  and compare strict field validation with the historical JSON loader.

- [x] Reject unknown nested fields in policy specification imports,
  as the historical JSON loader did. The installed R package also
  round-trips a one-observation specification, preserving the count
  without including data columns in the export. All 24 Rust boundary
  tests and focused Clippy passed.
- [x] Verify policy specification round trips in the rebuilt Python
  wheel and Julia. Each retains the one-observation count, omits raw
  data, and rejects an unknown nested causal model field.

## Dataset ingestion and provenance

- [x] Restore provenance record validation in Rust. The native request
  normalizes required text and rejects unknown license markers and
  unsupported provenance schemas. All 25 boundary tests and focused
  Clippy passed.
- [x] Restore content checksum verification, ingestion and dataset
  diagnostics, then qualify installed Python, Julia and R consumers.

- [x] Compute and verify SHA-256 dataset digests over sorted Rust JSON.
  Provenance extraction time and the checksum itself are excluded.
  The boundary test checks an independent digest and rejects a changed
  data value under a retained checksum. All 26 tests and focused
  Clippy passed. The direct dependency reuses locked sha2 0.11.1.
- [x] Qualify historical Python digest compatibility for non-ASCII
  strings and numeric encodings, and verify installed consumers.

- [x] Add an explicit historical Python JSON checksum encoding for
  finite JSON datasets. The fixture matches CPython 3.15.2 for Unicode
  keys and surrogate pairs, DEL, signed zero, integer versus float
  spelling, and exponent formatting. Rust encoding remains the default.
  All 27 native boundary tests and focused Clippy passed.
- [x] Verify a stored CPython dataset digest in the rebuilt Python
  wheel, Julia and installed R. The same fixture contains fractional
  numbers and Unicode text. All three accept the retained digest and
  reject the same changed numeric value. Broader float encoding is
  covered by the sampled native comparison below.

- [x] Correct decimal midpoint rounding in Python-compatible dataset
  digests. A seeded raw-bit sample found six disagreements in the
  initial 19,986 finite values. The corrected encoder matched CPython
  3.15.2 on 199,886 finite values through the rebuilt C ABI. The Rust
  regression retains a failing midpoint from the original sample.
  All 27 boundary tests and focused Clippy passed.

- [x] Restore adoption dataset diagnostics for column alignment,
  missingness, ordered and unique time keys, monotonicity, units,
  share bounds and denominators. The Rust fixture reports five
  independent failures in one dataset. All 28 boundary tests and
  focused Clippy passed.
- [x] Restore and verify diagnostic reports for all five historical
  dataset kinds across installed consumers. See the cases below.

- [x] Restore substitution diagnostics for aligned, rectangular share
  matrices, numeric entries, share units and total market share.
  The Rust regression rejects a combined share above one even when
  each product share is valid. All 29 boundary tests and focused Clippy passed.

- [x] Restore competition panel and policy timing diagnostics. Time can
  repeat across distinct units, products or event labels. Composite
  key duplicates fail and unsorted policy events warn. All 30 native
  boundary tests and focused Clippy passed.

- [x] Restore network edge dataset diagnostics for endpoints, ordered
  duplicate pairs, units, and nonnegative numeric weights. The fixture
  catches both a duplicate ordered pair and a negative weight while
  allowing the reverse pair. All 31 native boundary tests and focused
  Clippy passed.
- [x] Verify all five dataset report kinds in the rebuilt Python wheel,
  Julia and installed R. Reports agree on valid adoption and competition
  rows, excessive total share, unsorted policy timing and negative
  edge weights. R preserves singleton columns for this operation.

- [x] Connect native diagnostics and provenance verification in the
  structured ingestion request. Successful requests return a stamped
  dataset artifact and report. The fixture verifies re-ingestion,
  rejection of a changed dataset with a retained checksum and rejection
  of a failed diagnostic. All 32 boundary tests and focused Clippy passed.
- [x] Restore Polars local table ingestion and qualify the structured
  artifact handoff in Python, Julia and R on the fixtures below.

- [x] Restore Python Polars local ingestion for CSV, Parquet and Arrow
  IPC. The rebuilt wheel produced the same checksummed artifact from
  all three formats and a LazyFrame for each of the five dataset kinds.
  Rust owns validation and checksums.
- [x] Verify the structured artifact in Julia and R, source provenance
  enrichment, and schema version and product label checks.
- [x] Qualify artifact metadata and diagnostic details on fixtures for all
  five historical dataset kinds. Python 3.15.2 ingested local Polars CSV
  tables with Unicode provenance metadata. Julia and installed R
  re-ingested each artifact and reproduced the complete dataset, digest,
  provenance and validation report. Policy timing retained its
  event_time_order warning. R used raw JSON to retain numeric tokens.
- [ ] Complete broader dataset input and release qualification.

- [x] Verify re-ingestion of a Python-produced dataset artifact in Julia
  and installed R. Both reproduce the artifact and reject a changed
  value under its retained checksum. R exposes kernel_call_json to keep
  the original numeric tokens because list serialization can rewrite
  whole-valued floats as integers.

- [x] Restore local source provenance enrichment. The Python loader
  adds a load step and records the supplied path and normalized format,
  preserving caller metadata. CSV and Parquet reads with equal rows now
  carry different digests because their provenance differs. The rebuilt
  wheel verified both digests through the Rust checksum operation.

- [x] Enforce dataset schema 1.0 when explicitly supplied and validate
  substitution product labels against the share matrix width. Installed
  R verifies singleton substitution and edge artifacts, including the
  distinction between the edge source column and provenance source.
  All 32 Rust boundary tests and focused Clippy passed.

- [x] Return failed diagnostic reports in ingestion errors so callers
  can identify rejected checks without submitting the dataset twice.
  The rebuilt Python wheel exposes the report on NativeError.
  All 32 boundary tests and focused Clippy passed.

- [x] Preserve ingestion diagnostic reports in Julia and R errors.
  Julia raises DatasetValidationError and R signals
  innovate_dataset_validation_error for failed reports. Both bindings
  retain the native invalid_request code and the failed monotonicity
  check on the same rejected dataset used in Python.

## Native scenario grid execution

- [x] Execute an ordered grid of model simulations through the Rust kernel.
  Send schema 2.0 with operation `execute_scenario_grid` and
  `payload.scenarios`. Each entry supplies a unique nonblank `id`,
  `model_key` and ordinary `simulation` payload. Results and model errors
  stay with their scenario IDs. The kernel runs sequentially and reuses
  the simulation payload's seed for stochastic engines.
- [x] Verify the scenario grid in the rebuilt CPython 3.15.2 wheel, Julia
  and freshly installed R. The same seeded Bass network with one resource
  slot returned completion ticks 6, 11 and 16 through each binding.
  All 33 native boundary, 9 Kairos and 22 network tests passed, as did
  focused Clippy.
- [ ] Restore historical scenario schema and registry coverage, comparison
  metrics, execution metadata and release parity.

- [x] Restore the five historical scenario specification shapes on native
  grid entries (e2c5f355). The optional specification is validated before
  execution and returned with the row. Model execution uses the explicit
  simulation payload. Rust tests cover valid and invalid inputs for all five
  types. A network specification round-tripped through a rebuilt Python
  3.15.2 wheel, Julia and installed R. All 34 native boundary tests and
  focused Clippy passed.

- [x] Restore bounded parallel scenario grids (08741eea). The optional
  workers setting accepts integers from 1 to 32 and defaults to 1. Scenario
  and model batches share ordered chunk scheduling. Python 3.15, Julia and
  installed R each returned identical serial and parallel responses for a
  seeded Kairos resource simulation alongside an unsupported-model error.
  All 34 native boundary tests and focused Clippy passed.

- [x] Compare scalar grid results by scenario ID (92ededf2). Optional
  comparisons take baseline, alternative and threshold. Compare only
  identical strictly increasing time grids with one value per sample.
  Callers supply paths in common units. Return alternative minus baseline,
  percentage change (null when baseline is zero), and the first sampled
  time at or above a supplied threshold (null when never reached).
  Grid mismatch errors preserve the executions. Python 3.15, Julia and R
  reproduced a difference of 50 and a first alternative threshold sample
  at time 1. All 35 Rust boundary tests and focused Clippy passed.
- [ ] Qualify ranking, uncertainty bounds and richer comparison contracts.

- [x] Restore opt-in execution metadata (eb7ada2a). Set
  include_execution_metadata to retain the exact simulation payload,
  optional scenario notes, kernel version and response metadata, Unix
  start time and monotonic elapsed seconds. The default omits wall-clock
  fields to preserve deterministic replay comparisons. Python 3.15,
  Julia and R returned notes, timing and the Rust runtime marker for the
  same Logistic case. All 35 boundary tests and focused Clippy passed.

- [x] Restore historical scenario family listing and keyed lookup via
  discover_scenarios (7425c1ee). Rust publishes the five schema families
  with current transport, binding and execution requirements. An unknown
  key returns the available families. The rebuilt Python 3.15 wheel,
  Julia and installed R all listed five families and resolved network to
  the same row. All 36 Rust boundary tests and focused Clippy passed.

- [x] Restore final-adoption ranking and incremental effects (c2a533b3).
  Scalar comparisons also accept an explicit relative margin to reproduce
  the historical fixed-width heuristic. Bounds are labelled as a fixed
  relative margin, with no statistical coverage claim. Python 3.15, Julia
  and installed R agreed on an alternative score of 100 and baseline
  bounds of 45 to 55 with a 10% margin. All 36 boundary tests and focused
  Clippy passed.

- [x] Summarize paired replicate effects (a8bee915). A comparison can
  supply pairs of unique scenario IDs and an interval_mass in (0, 1).
  Successful scalar paths must share the same time grid and use common
  units. The result includes final-sample differences, the median and a
  central empirical interval using linear quantiles. Callers design and
  seed the replicates. These quantiles describe observed effects.
  Python 3.15, Julia and installed R agreed on [-6.25, 31.25] for the
  two-pair analytic fixture. All 37 boundary tests and focused Clippy passed.
- [ ] Qualify stochastic ensemble design and sampling-based inference.

- [x] Qualify a seeded Kairos ABM scenario ensemble against the
  innovation-only binomial mean (15b8ee5f). For 128 pairs of 100 agents,
  the observed mean final effect was 63.25 versus the analytic 63.21.
  Serial and four-worker Rust responses were identical. Python 3.15,
  Julia and installed R reproduced every paired difference and both
  empirical bounds (approximately 56 to 73). The R comparison used
  17-digit formatting to preserve decoded numeric precision.
  All 10 Kairos tests and focused Clippy passed.

- [x] Report the paired mean effect and its standard error (e9250428).
  The standard error uses sample variance and assumes independent
  replicate pairs. Python 3.15, Julia and installed R reproduced a mean
  of 12.5 and standard error of 37.5 for effects 50 and -25. All 37
  boundary tests, 10 Kairos tests and focused Clippy passed.

- [x] Enforce a common declared time basis for pair and ensemble
  comparisons (972fada2). If any scenario has a specification, every
  compared scenario must agree on time unit and reference year. Matching
  comparisons report that basis. Without specifications, callers supply
  a common basis. Python 3.15, Julia and installed R all rejected a
  year-versus-month comparison while preserving both execution results.
  All 37 Rust boundary tests and focused Clippy passed.

- [x] Add an opt-in seeded percentile bootstrap for the paired mean
  effect (38eafde3). Resampling draws entire pair differences with
  replacement and assumes independent pairs. The grid retains the
  empirical interval separately and bounds bootstrap workload.
  Python 3.15, Julia and installed R reproduced [12.5, 50] with
  seed 42 and 1,000 resamples. All 47 targeted Rust tests and focused
  Clippy passed. Broader coverage and stochastic designs remain open.

- [x] Check paired percentile resampling against a finite exact
  distribution (16623e59). Effects of -25 and 50 yield four equally
  likely ordered bootstrap means of -25, 12.5, 12.5 and 50. The seeded
  10,000-draw run recovered the central 90% endpoints of -25 and 50.
  Identical effects of 50 produced a point interval. The focused Rust
  regression and Clippy passed. This checks conditional resampling
  behavior. Repeated-sampling coverage remains open.

- [x] Export historical scenario artifact fields and a native replay
  request for successful executions when metadata is enabled (aa2a748c).
  The envelope records model, version, UTC timestamp, notes, optional
  specification and the effective seed reported by stochastic engines.
  Python 3.15, Julia and installed R each replayed a seeded Kairos ABM
  artifact to an identical result. All 37 boundary tests and focused
  Clippy passed. Legacy artifact import and cross-version replay remain
  open.

- [x] Import legacy scenario artifact envelopes (bd0dd498). Validate the
  historical specification and offset-bearing timestamp, preserve model,
  version, seed and notes, and report replay unavailable when the
  original envelope has no simulation payload. Python 3.15, Julia and R
  round-tripped the same legacy Bass artifact. All 38 Rust boundary tests
  and focused Clippy passed. Cross-version replay qualification remains open.

- [x] Preserve the validated legacy artifact's JSON values (db78ac51).
  Validation no longer reserializes integer-valued scenario fields as
  floats. The Rust boundary regression also retains the maximum unsigned
  64-bit seed. All 38 boundary tests and focused Clippy passed. Binding
  numeric representation limits for large seeds remain to be qualified.

- [x] Qualify lossless legacy seed transport across all bindings
  (5f67e986). Python 3.15 and Julia decoded the maximum unsigned
  64-bit seed exactly. Installed R preserves that seed as a JSON
  integer token with kernel_call_json(request, raw = TRUE), while
  retaining native error handling. The focused installed R regression
  passed. Default R decoding uses ordinary R numeric types.

- [x] Import native scenario artifacts and check replay compatibility
  (3681c8a8). Matching schema and kernel versions permit replay of
  the embedded request. Changed versions retain the artifact with a
  replay-unavailable reason. Inconsistent envelope and replay metadata
  produce an invalid request. Python 3.15, Julia and R verified the
  same-version and changed-version cases. All 38 Rust boundary tests
  and focused Clippy passed. Numerical parity across releases remains open.

- [x] Reject grid options injected into an embedded single-scenario replay
  request. Native imports accept the emitted scenarios-only payload.
  A regression covers a worker override that previously passed import
  but failed when replayed. All 38 Rust boundary tests and focused
  Clippy passed.

- [x] Require agreement between the artifact seed and the replay seed.
  A recorded seed requires a matching unsigned integer in replay
  options. Deterministic artifacts may omit the seed from both places.
  All 48 Rust boundary and Kairos tests and focused Clippy passed.

- [x] Execute a scenario grid on a shared Arrow IPC time column in Rust
  and from Python Polars DataFrame or LazyFrame inputs. Reuse the native
  time validator and reject ambiguous JSON time inputs. Results retain
  the JSON grid envelope for heterogeneous outputs. The rebuilt Python
  3.15 wheel returned the analytic Logistic value of 50. All six tabular
  tests and focused Clippy passed after removing a stale assertion that
  treated the supported rate quantity as invalid.
- [x] Qualify Arrow transport for grid-level comparisons.

- [x] Expose scenario Arrow IPC input through the C ABI to Julia and R.
  Both returned the analytic Logistic value of 50 and rejected malformed
  IPC from the shared fixture. The rebuilt installed R package, locked
  FFI test and focused Clippy passed. Result transport remains JSON.

- [x] Export ordered scenario execution rows as Arrow IPC in Rust and
  expose them as a Polars frame in Python. Index columns retain scenario
  ID, model and status. The execution_json column preserves nested
  results, artifacts and errors, including full-width seeds. Grid-level
  comparisons remain in the JSON envelope. All seven tabular tests,
  focused Clippy and the rebuilt Python 3.15 wheel smoke test passed.

- [x] Expose ordered execution-row Arrow IPC export to Julia and R.
  Polars decoded both outputs with identical ordered execution rows,
  including a maximum-width seed and a failed scenario. Both bindings
  rejected an invalid envelope. Locked FFI tests and Clippy passed.
  Grid-level comparisons remain in the JSON envelope.

- [x] Export grid comparison rows as Arrow IPC from Rust, Python, Julia
  and R. Ordered indices, comparison kind and status are typed columns.
  Nested pair and ensemble outcomes remain intact in comparison_json.
  The three installed consumers produced equal Polars tables for a pair
  result and an ensemble error. Eight tabular tests and Clippy passed.

- [x] Translate baseline Bass specifications when apply_specifications
  is enabled. Kairos ABM and DES use the declared market size and the
  initial adoption fraction times the explicit agent population, rounded
  to the nearest agent with halves rounded up. Conflicting explicit fields
  fail. Replay stores resolved inputs. All 49 targeted Rust tests and
  Clippy passed. Other scenario families and binding qualification remain
  open.

- [x] Advertise opt-in Bass baseline translation in native discovery.
  Python 3.15, Julia and installed R verified both Kairos ABM and DES:
  25% initial adoption in 20 agents gave five initial adopters and
  50 adoption units in a market of 200. All 39 boundary tests and
  focused Clippy passed. Other families remain open.

- [x] Translate Bass intervention specifications into additive Kairos
  innovation or imitation events. Reject unsupported intervention types,
  events outside the declared horizon and competing explicit event lists.
  Discovery advertises both supported delta types. The Rust fixture pauses
  innovation at time zero and replays the resolved event exactly. All 50
  targeted Rust tests and focused Clippy passed. Binding qualification
  for intervention translation remains open.

- [x] Qualify translated Bass interventions in the rebuilt Python 3.15
  wheel, Julia and installed R. Innovation and imitation cancellations
  both held adoption at five agents through time five in Kairos ABM and
  DES. Each binding replayed all four resolved artifacts exactly.

- [x] Translate custom network Bass specifications for Kairos. Derive
  market size and node count from the specification, check initial mean
  degree and rounded adoption count against the explicit graph and agent
  identities, and retain resolved inputs for replay. For directed graphs,
  average degree means average incoming degree. All 63 targeted Rust tests
  and Clippy passed. Binding qualification and generated topology remain open.

- [x] Qualify custom network specification translation in Python 3.15,
  Julia and installed R. A four-node graph with one edge and one initial
  adopter returned 50 adoption units in a market of 200. All three
  bindings replayed the resolved artifact to an identical result.

- [x] Translate competition specifications to the multi_product model.
  Allocate market size times initial adoption in the declared competitor
  order using initial shares. Derive n_products, reject conflicting
  constructor fields and retain resolved initial values for replay.
  Product-specific dynamics and market potentials remain explicit. All 57
  targeted Rust tests and Clippy passed. Binding qualification remains open.

- [x] Qualify competition specification translation in Python 3.15,
  Julia and installed R. Each returned 25 and 75 initial adopters in
  competitor order and replayed the resolved artifact exactly.

- [x] Generate undirected degree-two ring networks from a Kairos Bass
  scenario specification. Reject conflicting explicit edges, weights or
  directed mode. Replay retains generated edges. All 64 targeted Rust
  tests and focused Clippy passed. Other generators remain open.
- [x] Define substitution-rate dynamics before translating historical
  substitution specifications. Norton-Bass exposes generation diffusion
  parameters, with no direct replacement-rate parameter.

- [x] Qualify generated ring topology in Python 3.15, Julia and installed
  R. Each returned the expected four edges in order and reproduced its
  original result when replaying the resolved artifact.

- [x] Add an explicit fixed_pool_replacement scenario model. Interpret
  substitution_rate as a constant one-way replacement hazard per scenario
  time unit. The analytic path conserves the initial adopter pool and
  reports incumbent and entrant levels in that order. The scenario grid
  retains resolved inputs for replay. All 43 boundary tests and Clippy
  passed. Binding qualification and broader substitution designs remain open.

- [x] Qualify fixed-pool substitution in Python 3.15, Julia and installed R.
  All three matched the analytic incumbent level of 100 exp(-1), conserved
  the pool of 100 and replayed the resolved artifact exactly.

- [x] Generate seeded random fixed-edge networks for Kairos scenarios.
  Uniformly sample distinct undirected edges without replacement using
  topology_seed, independently of the simulation seed. Require a feasible
  exact edge count from mean degree. Store sorted resolved edges in replay
  requests. All 65 Rust boundary and network tests and Clippy passed.
  Cross-binding qualification and other topology families remain open.

- [x] Qualify seeded random fixed-edge graphs in Python 3.15, Julia and
  installed R. Topology seed 77 produced edges (0,1), (0,2), (0,3) and
  (1,2) in all three. Resolved replay requests agreed across bindings and
  each replay reproduced its original seeded contagion result exactly.

- [x] Generate seeded scale_free networks by clique-seeded preferential
  attachment. Each new node selects distinct prior nodes in proportion
  to their degrees. Accept only mean degrees compatible with 1 to 32
  links per new node and retain the resolved graph for replay. All 65
  targeted Rust tests and Clippy passed. Binding qualification remains open.

- [x] Qualify scale_free topology in Python 3.15, Julia and installed R.
  All three resolved the same seven distinct edges for a five-node graph
  with two attachments per new node and topology seed 77. Each replay
  reproduced the original seeded contagion result exactly.

- [x] Preserve submitted simulation inputs alongside resolved inputs in
  opt-in execution metadata when applying specifications. The regression
  verifies that the topology seed is retained in submitted_simulation
  and excluded from the resolved kernel payload. All 43 boundary tests
  and Clippy passed.

- [x] Preserve submitted inputs inside standalone translated artifacts.
  Same-version import regenerates the resolved request and checks exact
  JSON value agreement. The regression rejects a missing topology seed.
  Keep JSON numeric types intact during transport, using raw R JSON where
  needed. All 43 Rust boundary tests and focused Clippy passed.

- [x] Qualify translated artifact provenance in the rebuilt Python 3.15
  wheel, Julia and installed R. Python and R exports agreed on submitted
  inputs and resolved replay requests. All three accepted intact artifacts.
  Julia and R rejected removal of the topology seed. R used raw JSON to
  preserve integer and floating-point representations through import.

- [x] Restore deterministic threshold sensitivity in Rust, including
  inclusive counts, threshold order, proportions and outcome summaries.
  All 44 boundary tests and Clippy passed. Other report families and
  binding qualification remain open.

- [x] Restore parameter perturbation reports for final scalar native
  predictions. Perturb one explicit parameter at a time and report absolute
  change, relative change and elasticity. The Logistic fixture verifies
  independent perturbations. All 45 Rust boundary tests and focused
  Clippy passed. Binding and wider report qualification remain open.

- [x] Restore assumption sensitivity for explicit constructor options.
  Evaluate each alternative against the same final scalar baseline.
  The Logistic fixture moves an event across the final observation and
  checks both pre-event and post-event outcomes. All 46 Rust boundary
  tests and focused Clippy passed. Binding qualification remains open.

- [x] Restore intervention timing sensitivity for explicit scalar time
  options. Evaluate signed offsets against a fixed baseline and return
  times, outcomes and absolute changes. Reject negative or nonfinite
  times. All 47 Rust boundary tests and focused Clippy passed. Binding
  qualification remains open.

- [x] Qualify threshold, parameter, assumption and intervention timing
  sensitivity in the rebuilt Python 3.15 wheel, Julia and installed R.
  All four full result objects agreed across bindings and matched the
  inclusive threshold counts and analytic Logistic outcomes.

- [x] Restore the four native explainability summaries: weighted adoption
  contributions, focal competition pressure, first substitution threshold
  indices and policy decomposition. Equal-magnitude policy components
  resolve by name for deterministic JSON-object transport. All 48 boundary
  tests and Clippy passed. Binding and combined-report qualification remain open.

- [x] Restore ordered composition of sensitivity and explainability
  blocks via combine_report_blocks. Preserve nested values and compute
  the combined determinism flag. Missing flags count as false and
  nonboolean flags are rejected. All 49 Rust boundary tests and focused
  Clippy passed. Installed binding qualification remains open.

- [x] Qualify all four explainability summaries and ordered report
  composition in the rebuilt Python 3.15 wheel, Julia and installed R.
  The complete combined report matched across the three bindings.

- [x] Restore decision-report schema 1.0 JSON export in Rust. Preserve
  report blocks and evidence references, derive interpretation and claim
  counts, and enforce historical claim wording and causal assumption
  checks. The title is also checked for unsupported wording. All 50 Rust
  boundary tests and Clippy passed. Markdown and binding tests remain open.

- [x] Restore Rust Markdown decision report export from the validated JSON
  contract. Include claim boundaries, evidence references, optional blocks
  and claim-safety metadata. Escape Markdown control characters in user
  text and render line breaks as spaces. All 51 Rust boundary tests and
  Clippy passed. Binding verification and release parity remain open.

- [x] Qualify schema 1.0 JSON and Markdown decision reports in the rebuilt
  Python 3.15 wheel, Julia and installed R. Both complete output objects
  matched across bindings. Each rejected a causal claim without an
  identification assumption. Release parity remains open.

- [x] Restore model-card discovery with all ten historical benchmark
  blueprints alongside the current Rust model capabilities. Historical
  case IDs are descriptive evidence and do not imply a native benchmark
  run. All 52 Rust boundary tests and Clippy passed. Corpus, runner,
  additional cards and binding qualification remain open.

- [x] Restore the four historical synthetic smoke cases and native
  fit/forecast execution. Discovery retains case IDs, model keys and
  dataset version. Runs return observations, native fit diagnostics and
  predictions from fitted state. Scalar cases match their generating
  curves. The competition case verifies execution and output shape. All
  53 Rust boundary tests and Clippy passed. Suite automation, bindings
  and release qualification remain open.

- [x] Restore native smoke-suite execution with optional ordered case IDs
  and bounded workers. Report successful and failed cases separately while
  retaining request order and each case diagnostic. The default suite ran
  all four historical corpus cases. All 54 Rust boundary tests and Clippy
  passed. Historical automation, binding and release qualification remain open.

- [x] Qualify native benchmark suite and model-card lookup in the rebuilt
  Python 3.15 wheel, Julia and installed R. All returned identical Bass
  card and ordered suite results for Logistic, Bass and an unknown case.
  The unknown case remained a row error while both fits succeeded.

- [x] Restore fast benchmark card freshness reporting. Resolve historical
  card case IDs against native dataset versions and flag missing references
  as errors. Bass resolves to 2026.04 and lock-in remains stale because its
  smoke case is absent. The focused Rust regression and Clippy passed.
  Corpus completion, scheduling policy and binding qualification remain open.

- [x] Add the missing lock_in_smoke case using the existing independent
  SciPy fixture under dataset version native-lock-in-1. Its native fit
  starts from the fixture parameters and reproduces both trajectories
  within 1e-5. All ten historical cards now resolve to five corpus cases.
  Four focused benchmark tests and Clippy passed. Scheduling, wider card
  coverage and release qualification remain open.

- [x] Qualify the lock-in smoke case and corpus freshness in Python 3.15,
  Julia and installed R. Complete results agreed across bindings. All ten
  historical cards resolved, and the maximum trajectory error against the
  independent lock-in reference was 2.60e-8.

- [x] Add benchmark CI policy metadata and suite selection. Fast runs
  select the three scalar cases. The two competition cases are scheduled
  cases. Validation reports policy counts, and explicit case IDs remain
  available as an alternative selector. The focused Rust test and Clippy
  passed. Hosted scheduling and binding qualification remain open.

- [x] Wire scheduled and manually dispatched Ubuntu CI to the native
  scheduled benchmark policy and retain the JSON report as a workflow
  artifact. Existing test gates remain enabled. The repo-local Python
  3.15 CLI ran both scheduled cases successfully from a rebuilt wheel.
  Actionlint and Ruff passed. Hosted execution remains unverified.

- [x] Describe all six native models without historical benchmark cards.
  Include model-specific limitations and the existing Rust test target.
  Leave benchmark case IDs empty where no runnable corpus case is assigned.
  Discovery now returns a description for all 16 models. The focused
  Rust test and Clippy passed. New native benchmark cases remain open.

- [x] Add a runnable network diffusion benchmark with two analytic Bass
  node trajectories and zero spillover. Link it to the native model card
  and resolve its dataset version in corpus validation. Classify it as
  manual. Five focused Rust tests and Clippy passed. Nonzero spillover
  benchmark coverage and installed binding qualification remain open.

- [x] Qualify network_diffusion_smoke in Python 3.15, Julia and installed R.
  Complete benchmark results agreed across all three bindings. Across all
  26 samples and both nodes the maximum absolute error against the analytic
  Bass trajectories was 2.814e-12. This case uses zero spillover.

- [x] Add a fixed-parameter network spillover benchmark with a weighted
  directed graph and isolated node. Compare all three predicted paths
  with analytic Bass and spillover values. Link the case to the model card
  and keep it under the manual policy. Five focused Rust tests and Clippy
  passed. Installed binding qualification remains open.

- [x] Qualify the fixed-parameter weighted spillover benchmark in Python
  3.15, Julia and installed R. Complete results agreed across bindings.
  Across all four time points and three nodes, maximum absolute error
  against the analytic reference was 1.422e-13.

- [x] Add a manual fixed-parameter policy hazard benchmark and link it
  to the native card. Check positive and negative events at their exact
  boundaries, exponential decay and the cumulative maximum against an
  analytic Logistic reference. Five focused Rust tests and Clippy passed.
  Binding qualification and benchmark fitting coverage remain open.

- [x] Qualify policy_hazard_reference in the rebuilt Python 3.15 wheel,
  Julia and installed R. Full results agreed across bindings. All five
  predictions matched the analytic reference exactly on this host. This
  qualifies fixed-parameter prediction, with fitting coverage still open.

- [x] Add policy_hazard_fit to the native corpus and card. Fit a synthetic
  Bass path through the policy wrapper, keep the supplied decay fixed,
  and check convergence and trajectory recovery from saved state. Five
  focused Rust tests and Clippy passed. Binding qualification and
  estimation of intervention effects remain open.

- [x] Qualify policy_hazard_fit in Python 3.15, Julia and installed R.
  Complete fit and forecast results agreed across bindings. Decay stayed
  at 0.3 and the base fit converged. The maximum absolute trajectory error
  over 26 samples was 2.796e-12.

- [x] Add a fixed-parameter latent-process benchmark and link it to the
  native card. The analytic reference checks residual interpolation,
  endpoint extension and cumulative adoption across seven samples. Five
  focused Rust tests and Clippy passed. Binding and latent fitting
  benchmark qualification remain open.

- [x] Qualify latent_process_reference in the rebuilt Python 3.15 wheel,
  Julia and installed R. Complete benchmark results agreed across the
  three bindings and all seven predictions matched the analytic reference
  exactly on this host. Latent fitting benchmark coverage remains open.

- [x] Add and qualify latent_process_fit. The benchmark checks the fitted
  residuals against an independent smoothing recurrence and replays predictions
  from saved state. Five focused Rust tests, formatting and Clippy passed.
  The rebuilt Python 3.15.2 wheel and Julia returned identical results.
  Installed R matched within 1e-14 after JSON serialization. This benchmark
  covers the synthetic fit with fixed smoothing. Broader historical-model
  qualification and richer Kairos network/resource workflows remain open.

- [x] Add and qualify regime_switching_reference. The analytic reference
  checks both logistic segments and their continuity adjustment at the
  fixed-time boundary. All five samples matched exactly in the native result.
  Five focused Rust tests and Clippy passed. The rebuilt Python 3.15.2 wheel
  and Julia returned identical results, and installed R matched within 1e-14
  after JSON serialization. Regime fitting and changepoint selection still
  need coverage in the benchmark corpus.

- [x] Add and qualify regime_switching_fit. The synthetic logistic trajectory
  reproduces the existing independent detector fixture's breakpoint sequence,
  selects index 5 for the two-segment fit, and replays the fixed-time forecast
  from saved state. Maximum absolute trajectory error was 2.963e-11 over 30
  samples. Five focused Rust tests, formatting and Clippy passed. The rebuilt
  wheel loaded with Python 3.15.2 and Julia agreed exactly. Installed R agreed
  within 1e-14 after JSON serialization. This exercises detector parity on a
  smooth curve. Scientific validation on observed regime changes remains open.

- [x] Qualify the Kairos resource reopening and deadline tie. A seeded
  three-agent network keeps active service during closure. On reopening,
  it admits the higher-priority waiter before both deadlines expire on
  that tick. The other waiter withdraws and capacity is released on
  completion. All 23 Rust network tests and focused Clippy passed.
  Python 3.15.2, Julia and installed R agreed on counts, requests,
  completions, withdrawals and resource metadata. The test isolates
  resource behavior with imitation disabled.

- [x] Add and qualify hierarchical_group_fit. Fit two synthetic logistic
  groups and compare saved-state forecasts and the aggregate to analytic
  trajectories. Maximum group error was 2.125e-12 across 50 values. Five
  focused Rust tests and Clippy passed. Python 3.15.2 and Julia returned
  identical results. Installed R agreed within 1e-14 after serialization.
  This exercises independent group fits with centered offsets. Partial
  pooling and uncertainty remain separate obligations.

- [x] Add and qualify mixture_reference. Check the native weighted trajectory
  against a closed-form sum of two logistic components at five times. The
  native result matched exactly on this host. Five focused Rust tests and
  Clippy passed. The rebuilt Python 3.15.2 wheel and Julia returned identical
  results, and installed R agreed within 1e-14 after serialization.
  Gaussian mixture fitting remains open in the benchmark corpus.

- [x] Add and qualify mixture_gaussian_fit. Fit 50 alternating synthetic
  observations from two logistic components with perturbed capacity starts.
  The benchmark verifies likelihood monotonicity, capacity and weight
  recovery, and exact saved-state prediction replay. Five focused Rust
  tests and Clippy passed. The rebuilt Python 3.15.2 wheel and Julia
  returned identical results. Installed R agreed within 1e-14 after
  serialization. This case uses informed starts and deterministic samples.

- [x] Add and qualify composite_reference. Compare independent Bass and
  logistic trajectories with closed-form predictions at four times.
  Maximum absolute error across eight component values was 4.448e-9.
  Five focused Rust tests and Clippy passed. The rebuilt Python 3.15.2
  wheel and Julia agreed exactly. Installed R agreed within 1e-14 after
  serialization. The reference uses fixed initial values and zero coupling.
  Coupled cases and fitting remain open in the benchmark corpus.

- [x] Add and qualify composite_coupled_reference. With zero intrinsic
  growth and one directional interaction, the analytic trajectories
  are constant and linear. All eight component values and eight rate
  values matched exactly on this host. Five focused Rust tests and
  Clippy passed. The rebuilt Python 3.15.2 wheel and Julia agreed exactly,
  and installed R agreed within 1e-14 after serialization. This isolates
  directional coupling. Composite fitting still needs corpus coverage.

- [x] Add and qualify composite_fit. The synthetic single-component Bass
  case recovers parameters from perturbed starts, verifies one excluded
  conditioning row, and forecasts from saved state. Maximum error over
  25 values was 2.135e-10. Five focused Rust tests and Clippy passed.
  The rebuilt Python 3.15.2 wheel and Julia returned identical results.
  Installed R agreed within 1e-14 after serialization. Joint estimation
  of coupled components remains open.

- [x] Qualify composite_coupled_fit against a separate fixed-step RK4
  reference. Fit both logistic components and both signed interactions
  from perturbed starts. All active parameters recovered within 1e-5
  and saved-state prediction error across 50 values was at most 2.587e-9.
  Five focused Rust tests and Clippy passed. Python 3.15.2 and Julia
  results were identical. Installed R agreed within 1e-14 after JSON
  serialization. This is a noiseless, informed-start benchmark.

- [x] Accept Arrow IPC streams at the native Polars input boundary alongside
  files. A Polars 2.1.0 stream supplied the same scenario time grid through
  the rebuilt Python 3.15.2 extension, Julia and installed R. All eight
  tabular tests and focused Clippy passed. A separate read confirmed
  Arrow.jl 2.8.1 still returns no columns for the Python Polars IPC file,
  while it reads both the Polars stream and the tested Rust-written file.
  Direct Python Polars file readback in Arrow.jl remains open.

- [x] Provide a Polars-backed file handoff via Python export_ipc_file.
  The Rust writer emits a leading schema message. Python 3.15.2 Polars,
  Julia Arrow.jl 2.8.1 and R Arrow read the exported file and preserved
  Int32 counts, nullable Float64 and Unicode strings. All nine Rust
  tabular tests and focused Clippy passed. Direct Arrow.jl readback of
  files written by the Python Polars writer remains an upstream gap.

- [x] Accept local Arrow IPC streams in Python ingestion. Detect file versus
  stream framing for shared Arrow extensions and allow explicit stream
  suffixes. Two focused tests against the rebuilt Python 3.15.2 wheel
  passed. With the same path and rows, file and stream inputs produce
  the same validated, checksummed Rust artifact.

- [x] Retain Julia and R ingestion validation regressions in the binding
  suites. Focused checks passed for valid artifacts and structured
  monotonicity errors. Julia also verified exact re-ingestion of the
  stamped artifact. The Python local-stream checks are recorded above.

- [x] Add ordinal placement to pairwise final-adoption comparisons.
  Higher final adoption places first and equal values share first place.
  Historical score fields remain available for existing consumers.
  The focused Rust boundary test verifies the ordered and tied cases.
  Clippy passed. Installed binding qualification for placement remains open.

- [x] Qualify pairwise placement in the rebuilt Python 3.15.2 wheel,
  Julia and installed R. All three returned places 2 and 1 for unequal
  final adoption, and shared first place with the tie flag for equal
  values. Historical final-adoption scores matched the analytic values.

- [x] Add opt-in linear threshold timing for pair comparisons. Set
  interpolate_threshold with a threshold to report the first crossing
  between samples alongside the existing sampled timing. A path already
  at or above the threshold reports its first sample, and a path that
  never reaches it reports null. The focused Rust comparison test and
  Clippy passed. Binding qualification for this option remains open.

- [x] Qualify interpolated threshold timing through the rebuilt Python
  3.15.2 wheel, Julia and installed R. All three matched the analytic
  crossing between samples, returned the first sample for a path
  already above threshold, and returned null for unreached thresholds.
  The existing sampled timing agreed across the three bindings.

- [x] Add opt-in effective-seed validation for paired stochastic ensembles.
  require_distinct_pair_seeds requires matching seeds within each pair and
  distinct seeds between pairs. Invalid comparisons retain scenario results.
  The 128-pair Kairos regression verifies the accepted design, within-pair
  mismatch and cross-pair reuse. All ten Kairos tests and Clippy passed.
  Seed checks validate assignment only. Binding and inference qualification
  remain open.

- [x] Qualify paired seed assignment through the rebuilt Python 3.15.2
  wheel, Julia and installed R. Each accepted matched, distinct pair
  seeds and rejected both cross-pair reuse and within-pair mismatch.
  All six scenario executions remained available in each response.
  Sampling-based inference qualification remains open.

- [x] Measure repeated-sampling coverage for an innovation-only Kairos ABM
  design using the Python 3.15.2 wheel from f9f21916. Run 200 trials of 32
  paired runs with 20 agents, baseline p=0, alternative p=0.2, q=0 and
  horizon 5. Assign pair seed 32*trial+pair and use 1,000 bootstrap draws
  with seed 10000+trial. The nominal 90% paired percentile interval covered
  the analytic effect of 12.6424111766 in 176 of 200 trials (88%). Mean
  estimated effect was 12.609375 and mean interval width was 1.2284765625.
  Coverage for interacting or resource-constrained designs remains open.

- [x] Measure bootstrap coverage for capacity-constrained contagion.
  The Python 3.15.2 wheel from f9f21916 ran 200 trials of 32 matched-seed
  pairs on a star with one adopted hub and 19 leaves, p=0 in both arms,
  q=0 versus 0.05, capacity one, service time four, horizon five and
  1,000 ticks per unit. At most one new adoption can complete. The
  first request has total rate 0.95, so the analytic effect is
  1-exp(-0.95). Pair seeds were 20000+32*trial+pair, with 1,000 bootstrap
  draws at seed 30000+trial. The nominal 90% interval covered 183 of
  200 trials (91.5%). Grand mean was 0.62421875 versus the analytic
  0.6132589765. Mean interval width was 0.271390625. Coverage with
  multiple completions and feedback remains open.

- [x] Measure bootstrap coverage with multiple completions and feedback.
  Use a directed 0→1→2 chain with agent 0 adopted at time zero, no
  innovation, imitation 0 versus 0.5, one slot, service time one and
  horizon five at 1,000 ticks per unit. Each completion exposes the next
  agent. The analytic mean sums P(G₁ ≤ 4000) and P(G₁+G₂ ≤ 3000) for
  independent geometric delays with success probability 1-exp(-0.0005).
  Run 200 trials of 32 pairs, seed 40000+32*trial+pair, and 1,000 draws
  at seed 50000+trial using the Python 3.15.2 wheel from f9f21916.
  The nominal 90% interval covered 172/200 (86%). Mean effect was
  1.3140625 versus 1.306755628635 analytically. Both services completed
  in 2,841 of 6,400 alternative runs. Broader calibration remains open.

- [x] Replicate the feedback-chain coverage experiment with 500 new
  trials each at 32 and 64 pairs. Both sizes use pair seed
  60000+64*trial+pair and 1,000 bootstrap draws at seed 100000+trial.
  Coverage was 452/500 (90.4%) and 445/500 (89.0%), respectively.
  Mean interval widths were 0.38998125 and 0.281396875. The earlier
  86% result did not recur with new seeds at 32 pairs. The two sizes
  share their first 32 pairs per trial. Broader calibration is open.

- [x] Retain the repeatable feedback-chain coverage experiment in
  scripts/qualify_kairos_coverage.py. Run it with the qualified Python
  3.15 interpreter and a built Innovate wheel. It reproduced 452/500
  and 445/500 coverage and emits the analytic effect and mean interval
  widths as JSON. Ruff formatting and lint passed.

- [x] Retain the offset-aware weekly boundary fixture in the Julia and
  R suites. Both focused binding checks passed. Nine focused Rust
  preprocessing regressions passed. Broader SARIMA optimizer
  qualification remains tracked separately above.

- [x] Qualify stationary AR(1) likelihood on 500 seeded observations
  generated with coefficient 0.985 and initial value zero. The realized
  maximum is near 0.958185, reproduced by Rust within 2e-6 and log
  likelihood within 1e-8. R stats::arima reported nonconvergence near
  0.999775, so the independent reference maximizes the exact concentrated
  Gaussian likelihood with R optimize. Five focused Rust tests passed.
  Broader difficult mixed-order optimizer cases remain open.

- [x] Qualify a mixed-sign stationary ARMA(1,1) fit on 350 observations.
  The fixture uses R arima.sim with seed 419, AR 0.87 and MA -0.65.
  R stats::arima converged with exact ML and no mean. Rust matched
  estimated coefficients within 8.16e-6 and likelihood within 9.26e-8.
  Six focused Rust SARIMA tests and Clippy passed. Broader mixed
  seasonal and near-cancellation cases remain open.

- [x] Qualify stationary SARIMA(1,0,1)(1,0,0)[4] against R exact ML
  on 360 observations. Seed 482 generates a series with AR 0.6, MA
  -0.45 and seasonal AR 0.4. Rust matches estimated coefficients
  within 1.41e-5 and log likelihood within 8.67e-10. Six focused Rust
  SARIMA tests and Clippy passed. Near-cancellation remains open.

- [x] Qualify the mixed seasonal SARIMA fixture in Python 3.15.2, Julia
  and installed R through the native preprocessing request. Python and
  Julia agreed exactly. R agreed within 1e-10 on the coefficients,
  likelihood, and all 360 fitted and residual values.

- [x] Qualify one near-cancellation ARMA(1,1) fixture on 500 observations.
  Seed 872 uses generating AR 0.8 and MA -0.77. Of four R starts, only
  one converged. Rust agreed with it within 3.20e-5 on coefficients and
  1.20e-8 on log likelihood. Six focused Rust SARIMA tests passed.
  This qualifies one realized series, with broader optimizer coverage open.

- [x] Qualify the near-cancellation fixture through the Python 3.15.2
  wheel, Julia and installed R. Python and Julia returned identical
  outputs. R agreed within 1e-10 on coefficients, likelihood and all
  500 fitted and residual values. This verifies transport for the
  single near-cancellation series.

- [x] Isolate the automatic-start gap to the tested forced competition
  cases. At 200 iterations, the independent unforced multi-product and
  lock-in fixtures converge without parameter starts, with RSS below
  1e-10. The Python 3.15.2 probe and retained Rust regression passed.
  The forced convergence and held-out forecast gates remain open.

- [x] Test direction-preserving scaling for oversized least-squares
  steps on both forced competition fixtures. The rebuilt Python 3.15.2
  wheel used the existing 200-iteration budget and forecast oracle.
  Neither fit converged. Maximum held-out errors were 0.22728 for
  multi-product and 0.08580 for lock-in, both above 1e-3. Discard the
  experimental solver change. Forced initialization remains open.

- [x] Add a covariate-aware automatic start for forced multi-product
  and lock-in fits without parameter events. Fit approximate changes
  from trapezoidal ODE rates at observed endpoints, then use the result
  as an extra start for the existing trajectory solver. The new start
  converged in 195 and 185 trajectory iterations, respectively, with
  held-out errors 7.70e-8 and 4.06e-9. The rebuilt Python 3.15.2 wheel,
  focused Rust regression and Clippy passed. Julia and R verification
  for the new start remains open.

- [x] Qualify the forced automatic starts in Julia and installed R using
  the C ABI rebuilt from b38ea0bc. Both fixtures converged within the
  existing 200 trajectory iterations. Held-out errors were 7.70e-8
  and 4.06e-9 in both bindings, in line with Python 3.15.2. Automatic
  starts with parameter events remain a separate qualification case.

- [x] Extend the forced competition initializer to fixed parameter events.
  Exclude intervals crossing an interior event and evaluate an event at
  the right endpoint using its left-hand regime. The trajectory solver
  still integrates all observations. Python 3.15.2 and the Rust
  regression passed both independent event fixtures. Held-out errors
  were 9.72e-9 for multi-product and 7.15e-10 for lock-in. Focused
  Clippy passed. Julia and R event-start qualification remains open.

- [x] Qualify event-aware automatic starts in Julia and installed R
  with the C ABI rebuilt from 9351e28d. Both fixtures converged within
  the 200-iteration trajectory budget. Held-out errors were below 1e-8
  for multi-product and 1e-9 for lock-in in both bindings, consistent
  with Python 3.15.2. This covers the two fixed-event reference cases.

- [x] Accept fitted model state for counterfactual forecasts. Parameter
  overrides modify a copy of the saved parameters, preserving the
  baseline and state options. The native Logistic fixture returns
  50 and 60 and rejects an unknown override. Focused Rust and Clippy
  passed. Binding qualification remains open.

- [x] Qualify saved-state counterfactuals in the rebuilt Python 3.15.2
  wheel, Julia and installed R. Both Logistic samples show a 20% uplift
  after changing saved capacity from 100 to 120. Python and Julia
  agree exactly, and R agrees within 1e-12. The R list request omitted
  optional empty option objects.

- [x] Support named covariate history overrides on saved-state
  counterfactual forecasts. Preserve the saved time grid and apply
  replacement values only to the alternative. An analytic forced
  Lotka–Volterra fixture verifies the resulting trajectories and
  rejects a length mismatch. Focused Rust and Clippy passed.
  Installed binding verification remains open.

- [x] Qualify saved covariate history overrides in the rebuilt Python
  3.15.2 wheel, Julia and installed R. Complete results agreed exactly.
  The first Lotka–Volterra trajectory was 0.2319693167 at baseline
  and 0.1683698760 under the replacement, matching the analytic oracle
  within 1e-8. R used the raw JSON entry point.

- [x] Allow counterfactual replacement of a saved fixed parameter-event
  schedule. An empty array removes events from the alternative only.
  The analytic Lotka–Volterra test verifies the resulting forecast and
  rejects an unknown event parameter through native validation.
  Focused Rust and Clippy passed. Binding verification remains open.

- [x] Qualify event-schedule replacement and removal through Python
  3.15.2, Julia and installed R. Complete results agreed exactly and
  matched both analytic alternatives within 1e-8. R used raw JSON
  to retain the empty event array.

- [x] Qualify Innovate IPC file exports of the retained causal metadata
  and treatment contrast tables. Python 3.15.2 Polars round-tripped
  both files. Julia Arrow and R Arrow matched each existing stream
  column for type, values and nulls. Direct Arrow.jl readback of files
  emitted by the separate Python Polars writer remains open.

- [x] Schedule Kairos agent priority changes. Reorder existing waiters
  with their original queue entry times, apply new priorities to future
  requests, and leave active service and waiting deadlines in place.
  The regression verifies a promoted waiter completes next, active
  service is preserved, and malformed priority vectors fail.
  All 24 Rust network tests and focused Clippy passed. Binding
  qualification for this schedule field remains open.

- [x] Qualify scheduled agent priorities through the rebuilt Python 3.15
  wheel, Julia and installed R. Complete network results agreed exactly.
  The active agent finished at tick 6 as before, the promoted waiter
  finished at tick 11, and the other waiter finished at tick 16. R used
  raw JSON to retain the empty edge array.

- [x] Retain scheduled priorities through scenario specification
  translation and replay. The focused Rust regression verifies the
  effective priorities, original schedule in the replay request, and
  exact replay result for the custom network specification.

- [x] Verify FIFO ties after successive Kairos priority changes.
  Promote the later waiter at tick 4 and restore equal priority at tick 5.
  The Rust regression reproduces the original completion order and
  records both applied schedule ticks.

- [x] Schedule per-agent Kairos service durations for future admissions.
  A scheduled vector supersedes a previous scalar override. Reject a
  schedule entry that supplies both. The regression checks that active
  service retains its completion time and queued agents use the new
  durations. All 25 Rust network tests and focused Clippy passed.
  Binding verification remains open.

- [x] Qualify scheduled per-agent service durations in the rebuilt Python
  3.15 wheel, Julia and installed R. Complete responses agreed exactly.
  Active service finished at tick 6. Waiters used the new durations and
  completed at ticks 8 and 12. The scalar override was cleared. R used
  raw JSON to preserve the empty edge array.

- [x] Document scheduled Kairos priorities and per-agent durations in
  the native migration guide. Include queue tie behavior, deadline and
  active service preservation, and scalar/vector duration precedence.

- [x] Verify scenario translation and replay retain scheduled per-agent
  durations alongside priorities. The focused Rust test confirms the
  effective duration vector, retained schedule and exact replay result.

- [x] Reconcile the migration ledger with the qualified forced fits,
  five dataset artifact fixtures, Kairos ensemble evidence and native
  benchmark expansion. The development checker passes. The release
  checker still rejects the 38 required obligations pending acceptance.

- [x] Restore native declarative plugin manifest validation through
  validate_extension_manifests. Normalize historical stability aliases,
  validate the four historical extension points, and reject duplicate
  names in a submitted registry. Focused Rust and Clippy passed.
  Persistent registration and installed binding checks remain open.

- [x] Qualify native extension manifest validation in the rebuilt
  Python 3.15 wheel, Julia and installed R. All three returned identical
  normalized manifests, including the provisional tier and empty requires
  array. R used the raw JSON entry point. Registry lifecycle remains open.

- [x] Restore process-local extension manifest registration and listing.
  The native registry shares validation rules with batch validation,
  returns a name-ordered snapshot, and rejects duplicate registration
  without replacing the first manifest. Two focused tests and Clippy
  passed. Installed registration lifecycle checks remain open.

- [x] Verify process-local registration and listing on macOS through the
  rebuilt Python 3.15 wheel, Julia and installed R. Each returns the
  same normalized manifest on a subsequent call. Julia and R release
  their native library handles after calls, so persistence on loaders
  that unload the library remains a separate portability gate.

- [x] Retain native loader references across Julia and R JSON calls for
  process-local extension registration. Julia caches by resolved library
  path and R retains a loaded DLL reference. The retained lifecycle
  tests passed on macOS against Julia and freshly installed R. Hosted
  Linux and Windows runtime evidence remains open.

- [x] Classify plugin contracts as bounded native in the migration ledger.
  Manifest validation and the process-local registry have macOS binding
  evidence. Hosted lifecycle and extension-point execution remain open.
  Development policy passes and all 38 required obligations still gate release.

- [x] Restore seeded scalar case bootstrap refits through bootstrap_fit.
  Encode duplicated observations as integer fit weights, retaining the
  original ordered grid. Return baseline parameters, converged parameter
  draws and failed-draw counts. Focused Rust replay and Clippy passed.
  Interval summaries, broader model coverage and binding checks remain open.

- [x] Summarize scalar bootstrap draws with linear percentile intervals.
  Default to 95% mass, accept an explicit mass in (0,1), and return null
  for parameters with fewer than two converged draws. Reuse the scenario
  quantile implementation. Focused Rust and Clippy passed.

- [x] Verify scalar bootstrap through rebuilt Python 3.15, Julia and R.
  The seeded Logistic fixture converged on all eight draws and returned
  identical parameter draws and 90% percentile bounds in all bindings.
  The capacity interval collapsed to 100.000000000053 on noiseless data.
  R used raw JSON. Sampling coverage remains open.

- [x] Measure scalar case-bootstrap coverage on noisy Logistic data.
  Run 100 trials at times 0..23 with L=100, k=0.4, x0=10, Gaussian
  noise SD 0.5 from Python Random(2431), and 100 draws seeded 3000+trial.
  All observed values stayed positive. Nominal 90% capacity intervals
  covered 81/100, mean width 0.7462212, with 53/10000 failed refits.
  The initial SD 1 probe hit a baseline validation failure. This
  fixed-grid experiment does not establish general calibration.

- [x] Add centered residual resampling for fixed-grid scalar bootstrap.
  Resample centered residuals around converged baseline fitted values,
  retain the original time grid, and count failed synthetic refits.
  Focused Rust replay and interval checks and Clippy passed. This
  method assumes exchangeable residuals. Coverage and bindings remain open.

- [x] Verify fixed-grid residual bootstrap through rebuilt Python 3.15,
  Julia and R. All 30 refits converged on the perturbed Logistic fixture
  and complete outputs agreed exactly. The 90% capacity interval was
  [99.8197441, 100.3101662]. R used raw JSON. Repeated coverage remains open.

- [x] Compare centered residual bootstrap on the same 100 noisy Logistic
  trials and seeds as the case-bootstrap experiment. The residual method
  refused trial 81 because its baseline did not converge. Among the 99
  returned intervals, 85 covered L=100. Mean width was 0.69285245 and
  201 of 9,900 refits failed. Case resampling covered 81 of 100 with
  mean width 0.7462212. Preserve the differing denominators and report
  baseline refusal separately. Broader calibration remains open.

- [x] Expose baseline optimizer diagnostics and categorize failed scalar
  bootstrap draws as nonfinite synthetic samples, fit errors, or
  nonconverged refits. The focused Rust replay regression verifies
  that categories sum to the existing failed count. Clippy passed.

- [x] Exercise bootstrap diagnostics on noisy trials 0 and 81 in the
  rebuilt Python 3.15 wheel. Trial 0 converged at baseline and counted
  one nonconverged refit. Trial 81 reported baseline damping_limit and
  all 100 case refits converged. Failure categories summed to each total.

- [x] Verify noisy scalar bootstrap diagnostics in Julia and installed R
  against Python 3.15. Both trials reproduced the full Python result
  exactly, including baseline termination, failed-refit categories,
  parameter draws and intervals. R used raw JSON.

- [x] Restore parameter standard errors as sample standard deviations
  of converged scalar bootstrap draws. Scale before squaring and return
  null when fewer than two draws converge. The focused Rust test
  compares the result to the direct sample formula. Clippy passed.

- [x] Verify bootstrap standard errors in rebuilt Python 3.15, Julia
  and R on noisy Logistic trial 0. Complete results agreed exactly.
  Standard errors matched Python statistics.stdev on the converged
  draws within 1e-12 relative error. R used raw JSON.

- [x] Restore the bootstrap parameter Pearson correlation matrix.
  Constant columns yield null entries. Fewer than two parameters or
  draws yield a null matrix. Signed and constant-column unit checks,
  the native bootstrap regression, and Clippy across Rust tests passed.

- [x] Verify scalar bootstrap parameter correlations in rebuilt Python
  3.15, Julia and R. All three returned identical results on noisy
  Logistic trial 0 with 99 converged draws. Every matrix entry matched
  Python statistics.correlation within 1e-12 absolute error. R used raw JSON.

- [x] Reconcile bootstrap uncertainty as bounded native. Scalar case
  and residual refits, percentile intervals, standard errors and
  correlation have cross-binding evidence. The noisy Logistic
  experiments leave broader statistical calibration open. The development
  checker passes and release remains gated by 38 required obligations.

- [x] Qualify the native Polars IPC exporter through installed Voiage 3.1.0.
  The isolated Python 3.15.2 environment round-tripped the payoff table
  and returned EVPI 5.641720719927868 against 5.641720719927869.
  Record bounded native evidence while retaining broader release gates.

- [x] Qualify the Voiage handoff from the native Arrow scenario grid.
  Both Bass payoff draws and the Logistic alternative now come from
  ordered Rust scenario executions on a shared Polars grid. The installed
  Voiage 3.1.0 EVPI still matches the analytic reference within 1e-12.

- [x] Restore the remote execution envelope through a Rust in-process
  adapter. Validate both schemas, request size, scope label, retention
  and operation eligibility before dispatch. Return native provenance
  and correlation metadata. The scope is caller supplied. Focused
  Rust and Clippy passed. Binding and hosted transport remain open.

- [x] Verify the in-process remote execution envelope in rebuilt Python
  3.15, Julia and installed R. The native prediction and both policy
  denials agreed exactly after excluding elapsed milliseconds, which
  were nonnegative. R used raw JSON. Hosted authentication and
  transport remain open.

- [x] Restore remote execution contract discovery. Share operation,
  scope, retention and size constants with the native in-process
  dispatcher. Discovery identifies caller-supplied identity and
  exposes observability fields. Focused Rust and Clippy passed.

- [x] Verify remote contract discovery in rebuilt Python 3.15, Julia
  and R. Results agreed exactly on five eligible operations, a 1 MB
  limit and caller-supplied identity. Mark the local adapter bounded
  native in the migration ledger. Hosted qualification remains open.

- [x] Add native one-parameter Bayesian posterior quadrature.
  Evaluate the native scalar predictor with a bounded uniform prior
  and independent Gaussian observation errors at known noise scale.
  The Logistic capacity fixture recovers mean 100 and variance 4.
  Focused Rust and Clippy passed. Joint inference and binding checks remain open.

- [x] Verify Bayesian parameter quadrature through rebuilt Python 3.15,
  Julia and installed R. All 401 posterior masses and grid coordinates
  agreed exactly. The normalized Logistic capacity posterior recovered
  mean 100 and variance 4 within 1e-10. R used raw JSON.

- [x] Extend native Bayesian quadrature to two distinct parameter axes.
  Cap joint grids at 10,000 cells and return row-major posterior mass,
  axis coordinates, shape and marginal means. A 5-by-7 Logistic fixture
  matches independently computed Gaussian weights within 1e-12. Both
  focused tests and Clippy passed. Joint binding checks and MCMC remain open.

- [x] Verify two-axis Bayesian quadrature in rebuilt Python 3.15, Julia
  and installed R. All three returned identical row-major masses and
  marginal means for the 5-by-7 Logistic grid. An independent formula
  matched every mass and both means within 1e-12. R used raw JSON.

- [x] Return equal-tail credible intervals from discrete posterior
  grid marginals. Default to 95% and accept an explicit credible mass.
  The Logistic capacity interval agrees with the normal oracle to
  within one grid step. Both focused Rust tests and Clippy passed.

- [x] Verify Bayesian grid credible intervals in rebuilt Python 3.15,
  Julia and installed R. All three returned identical results. The
  Logistic capacity bounds [96.1097257, 103.8902743] matched independent
  cumulative-mass quantiles. R used raw JSON. Joint interval binding
  verification and MCMC remain open.

- [x] Verify joint Bayesian marginal intervals in Python 3.15, Julia and
  installed R. All three returned identical results on the 5-by-7 grid.
  Independently summing rows and columns reproduced the 95% bounds
  [92,108] for L and [0.2714285714,0.8428571429] for k. R used raw JSON.

- [x] Add seeded scalar random-walk Metropolis using the grid likelihood.
  Retain repeated states on rejection and report retained acceptance.
  A 4,000-draw Logistic chain replays exactly and recovers the known
  posterior mean within 0.25. Three focused Rust tests and Clippy passed.
  Joint sampling, convergence diagnostics and binding verification remain open.

- [x] Verify seeded scalar Metropolis through a rebuilt Python 3.15.2
  wheel, Julia and freshly installed R using the rebuilt Rust library.
  The full results agreed exactly on the 4,000 retained Logistic draws,
  posterior mean 100.13797081088984 and 2,543 acceptances (0.63575).
  R used raw JSON. Joint sampling and convergence diagnostics remain open.

- [x] Extend seeded Metropolis to 2–16 distinct parameter axes with
  per-axis bounds and proposal scales. Return one ordered vector per
  retained draw. The rebuilt Python 3.15.2 wheel, Julia and installed R
  returned identical 4,000-draw Logistic results for L and k, with
  61.3% acceptance. Independent means and transition counts agreed,
  duplicate axes were rejected, and the previous scalar response
  replayed exactly. Four focused Rust tests and Clippy passed.
  Multi-chain diagnostics and broader posterior calibration remain open.

- [x] Add classical split R-hat for equal-length posterior marginal
  chains. Report undefined diagnostics for zero within-split variance
  and reject unequal chain lengths. Python 3.15.2, Julia and R agreed
  exactly on the drifting five-draw fixture and matched sqrt(19/6).
  Five focused Rust tests and Clippy passed. Rank-normalized folded
  diagnostics and effective sample size remain open.

- [x] Add rank-normalized split and folded R-hat as an explicit method.
  Fold around the median of the retained split draws and report the
  maximum when both components are defined. On an odd-length scale
  mismatch fixture with ties, Python 3.15.2, Julia and R agreed exactly.
  The folded value 1.975023797027019 matched an independent base-R
  rank and normal-quantile calculation within 1e-12. Six focused Rust
  tests and Clippy passed. Effective sample size remains open.

- [x] Add bulk and 5%/95% tail effective sample sizes for split
  posterior marginal chains using the truncated initial monotone
  sequence. Report null for zero within-split variance. Python 3.15.2,
  Julia and R agreed exactly on a 320-draw autocorrelated fixture.
  Bulk ESS 23.66891870454829 and both tail estimates agreed within
  1e-10 with an independent base-R autocovariance calculation.
  Seven focused Rust tests and Clippy passed.

- [x] Combine native Metropolis and marginal diagnostics in
  bayesian_metropolis_chains. Require distinct explicit seeds and
  initial values for every sampled parameter. Return each chain and
  per-parameter split R-hat and effective sample size. Python 3.15.2,
  Julia and R agreed exactly on four dispersed Logistic capacity
  chains with 16,000 total draws, R-hat 1.000997 and bulk ESS 2796.26.
  Eight focused Rust tests and Clippy passed, including joint diagnostic
  ordering, deterministic replay and duplicate seed rejection.

- [x] Calibrate joint Metropolis against a 100-by-100 midpoint grid
  for Logistic capacity L and rate k. Use x0=2, observations 50 and 62
  at times 2 and 3, Gaussian noise SD 2, uniform bounds [80,120] and
  [0.1,0.9], and proposal half-widths 6 and 0.22. Seeds 41–44 start
  at (84,0.2), (92,0.7), (109,0.25) and (117,0.8), with 2,000 warmup
  and 10,000 retained draws each. Python 3.15.2, Julia and R returned
  identical grid and chain results. Independently calculated Gaussian
  weights matched every grid mass within 1e-14. Pooled means and
  correlation were 99.84115, 0.502064 and -0.76662 versus grid values
  99.77954, 0.502851 and -0.77315, within 0.2, 0.015 and 0.04.
  Both R-hats were below 1.004 and both bulk ESS values exceeded 2,000.
  Broader model and prior calibration remain open.

- [x] Share optional independent Gaussian priors between quadrature and
  Metropolis. The gaussian_priors map specifies mean and positive sd
  per sampled parameter. Bounds truncate each prior, and unspecified
  axes use uniform priors. Python 3.15.2, Julia and R agreed exactly on
  the Logistic capacity fixture with Gaussian prior mean 96 and sd 2.
  The grid recovered analytic posterior mean 98 and variance 2 within
  1e-10. The 4,000-draw chain returned mean 98.0372 and variance 1.8989.
  Nine focused Rust tests and Clippy passed.

- [x] Schedule Kairos retry policy changes. A replacement sets delay
  and retry budget for future balks and abandonments. Explicit null
  disables future retries, and already scheduled wakes keep their
  original time. Python 3.15.2, Julia and R agreed exactly on a cleared
  policy fixture with a wake at tick 6 and completion at tick 12.
  All 26 Rust network tests and focused Clippy passed.

- [x] Verify scheduled Kairos retry clearing through custom network
  scenario translation and artifact replay. The Rust regression passed.
  Python 3.15.2, Julia and R agreed on the replay request and reproduced
  the full simulation result, including the wake at tick 6 and second
  completion at tick 12. Document replacement and pending-wake semantics.

- [x] Trace Kairos service admissions with request time, queue wait,
  assigned duration and scheduled completion. The scheduled-duration
  fixture produced starts at ticks 1, 6 and 8 and waits of 0, 3 and 3
  ticks. Python 3.15.2, Julia and R agreed exactly. All 26 Rust network
  tests and focused Clippy passed.

- [x] Report Kairos occupied and configured slot-ticks, and split
  queue wait into admitted, abandoned and pending totals. Clip
  occupancy and pending waits to the observation horizon. On a
  closure fixture, Python 3.15.2, Julia and R returned identical
  results with five occupied slot-ticks, three configured slot-ticks,
  two abandoned wait ticks and one pending wait tick. All 27 Rust
  network tests and focused Clippy passed.

- [x] Verify Kairos resource metrics through scenario replay and Arrow
  export. Polars decoded execution files emitted through Python 3.15.2,
  Julia and R. Rows agreed after excluding per-run timestamps and
  elapsed time. All three reproduced the full closure simulation on
  replay. The focused Rust retry replay test also asserts resource
  accounting for its completed-service fixture.

- [x] Use the oldest Polars IPC compatibility level for native scenario
  execution and comparison files. R Arrow rejected the prior UTF-8 view
  columns. Python 3.15.2 Polars, Julia Arrow 2.8.1 and R Arrow read both
  rebuilt files and preserved their row counts and success flags. Both
  Rust scenario IPC tests and focused Clippy passed. The separate Python
  Polars file writer remains an open Julia interoperability item.

- [x] Expose IPC byte rewriting through Python's export_ipc_file.
  A rebuilt Python 3.15.2 wheel rewrote an existing Polars-written
  IPC file and accepted stream, eager and lazy inputs. Python Polars,
  Julia Arrow 2.8.1 and R Arrow read the resulting files and preserved
  Int32, nullable Float64 and Unicode strings. Julia still returned
  an empty schema for the original file. The focused Rust round-trip
  and Clippy passed.

- [x] Expose native IPC rewriting to Julia and R through the shared
  Rust byte-output ABI. Julia and a fresh R installation rewrote both
  a Polars file and a stream, and rejected malformed bytes. Python
  Polars verified all four outputs against the qualified Python
  exporter. Rust FFI tests and focused Clippy passed.

- [x] Reconcile Bayesian inference as bounded native in the migration
  ledger. Document the supported likelihood, priors, quadrature and
  multi-chain call. The published example executed in Python 3.15.2
  and recovered the reference posterior mean within 0.15. The
  development checker passed and release still lists all 38 gates,
  including Bayesian inference.

- [x] Profile supplied Logistic event times by refitting both branches
  at each candidate and selecting the lowest converged residual score.
  Save the selected fixed split in the fitted state. Python 3.15.2,
  Julia and R agreed exactly on the synthetic 24-observation fixture.
  Candidate 12 recovered the true split with RSS below 1e-18 against
  RSS above 5,000 at candidates 10 and 14. The focused Rust regression
  and Clippy passed. Broader model and event uncertainty work remains open.

- [x] Extend discrete event profiling to historical Bass and Gompertz.
  Reuse fixed-event fit semantics and save the selected split and fitted
  history. Python 3.15.2, Julia and R agreed exactly on the covariate
  Bass and analytic Gompertz fixtures. They selected times 10 and 2,
  respectively, with RSS below 1e-18. The Bass candidate at 11 remained
  visible as unconverged. All 11 mechanistic Rust tests and focused
  Clippy passed. Event uncertainty and broader model profiling remain open.

- [x] Require branch-level observation support for weighted event
  profiling. Candidates with fewer than three positive-weight rows on
  either branch are reported and excluded from selection. Python
  3.15.2, Julia and R agreed exactly on the Logistic fixture with
  eight zero-weight rows, excluding time 10 and recovering time 12.
  The focused Rust regression and Clippy passed.

- [x] Exercise historical event profiling through the installed Python
  Polars API. Expose candidate diagnostics as FitResult.event_profile.
  A rebuilt Python 3.15.2 wheel fitted a LazyFrame, recovered the split
  at 12, and predicted both branches and the boundary through the saved
  fit while preserving row IDs. The focused consumer regression passed.

- [x] Verify held-out Bass forecasts from an event-profile fit with
  retained covariate history. Python 3.15.2, Julia and R produced
  identical levels and rates at eight unseen times and matched the
  independent SciPy fixture within 0.001. The Python Polars consumer
  regression now covers candidate selection, training replay and
  held-out prediction alongside fixed-split fits. Four selected tests passed.

- [x] Profile one Norton–Bass parameter event by zero-based index.
  Refit base parameters at each candidate and retain the selected
  event list, fixed overrides, and other event times. Python 3.15.2,
  Julia and R agreed exactly on the SciPy fixture and selected 2.125
  with RSS 2.94e-14 while retaining the event at 5.375. The focused
  Rust regression and Clippy passed. Joint event and launch inference
  and uncertainty remain open.

- [x] Profile a later Norton–Bass launch in successive-generation
  fits. Retain the selected launch and the unchanged parameter events
  in fitted state. Python 3.15.2, Julia and R agreed exactly on the
  independent forcing fixture, selecting 3.125 with RSS 8.30e-19.
  Both focused profile tests and Clippy passed. Joint launch/event
  inference and uncertainty remain open.

- [x] Profile one Norton–Bass launch and one event jointly over up to
  16 candidate pairs. The fitted state retains the winning pair and
  the other event. Python 3.15.2, Julia and R returned identical
  results on the nine-pair fixture, recovering launch 3.125 and event
  2.125 with RSS 8.30e-19. Both focused Rust tests and Clippy passed.
  Continuous-time search, wider timing inference and uncertainty remain open.

- [x] Expose Norton–Bass launch and joint timing profiles on Python
  Polars FitResult. A rebuilt Python 3.15.2 wheel fitted the
  successive-generation LazyFrame, exposed all nine candidate pairs
  and forecast from the saved state within 0.001 of the SciPy
  reference. Both selected consumer tests and Ruff passed.

- [x] Verify held-out forecasting from each binding's saved joint
  Norton–Bass timing fit. Python 3.15.2, Julia and R returned identical
  generation forecasts using the retained forcing history. Both
  generations matched the independent SciPy fixture within 7.5e-10.

- [x] Extend indexed parameter-event profiling to Lotka–Volterra,
  complementary goods, lock-in and multi-product models. Python
  3.15.2, Julia and R agreed exactly on all four independent SciPy
  cases, recovering the first event at 2.125 while retaining the
  other event. Winning residual sums of squares were below 7e-18.
  The focused Rust test and Clippy passed. Overrides are held fixed.

- [x] Profile one existing indexed event override at fixed or candidate
  event times. The six-combination Lotka–Volterra fixture recovered
  time 2.125 and alpha1 override 0.4 with RSS 2.26e-20. Python 3.15.2,
  Julia and R agreed exactly. The focused Rust regression and Clippy
  passed. Continuous override estimation and uncertainty remain open.

- [x] Verify forecasts from each binding's saved Lotka–Volterra
  event-time and override fit. Python 3.15.2, Julia and R returned
  identical results using retained covariate history, and both
  held-out series matched the independent SciPy fixture within 3e-11.

- [x] Return the selected row and zero-based index in native timing
  profiles. Python 3.15.2, Julia and R agreed on the Lotka–Volterra
  event and override selection. The Rust regression, two Python Polars
  profile consumer tests and focused Clippy passed.

- [x] Report event selection counts and frequencies from scalar
  bootstrap refits. The seeded noisy Logistic residual fixture
  selected time 12 in all 12 successful refits, with exact agreement
  across Python 3.15.2, Julia and R. Focused Rust tests and Clippy
  passed. Frequencies describe stability on the candidate grid.

- [x] Verify paired-case event-selection frequencies when resampling
  loses branch support. Slice the noisy Logistic residual fixture to
  times 7–17 and use candidates 11, 12 and 13 with seed 42 and 80 draws.
  Python 3.15.2, Julia and R agreed exactly. Twelve refits returned fit
  errors, and the 68 successful refits selected the candidates 23, 35
  and 10 times. Frequencies sum to one over successful refits.

- [x] Extend paired case bootstrap to Norton–Bass and the competitive
  ecosystem models, with shared row weights across observed series.
  Report candidate frequencies for event, launch and joint profiles.
  Python 3.15.2, Julia and R agreed exactly on three joint Norton–Bass
  refits over four pairs. All selected launch 3.125 and event 2.125.
  The scalar regression and focused Clippy passed. Broader statistical
  calibration and multiseries residual resampling remain open.

- [x] Support centered multiseries residual bootstrap. Draw residuals
  by row after centering each series, and retain any conditioning
  observation. Python 3.15.2, Julia and R agreed exactly on three
  noisy Norton–Bass joint profile refits. Parameter estimates varied,
  and all refits selected launch 3.125 and event 2.125. The scalar
  regression and focused Clippy passed. Calibration remains open.

- [x] Expose the residual pool size and conditioned row count.
  A noisy Lotka–Volterra fit derived its initial condition from the
  first observed row. Python 3.15.2, Julia and R agreed exactly on
  all four refits with one conditioned row and 50 residual pool rows.
  The scalar regression and focused Clippy passed.

- [x] Bootstrap pointwise forecast intervals from refitted states.
  Accept up to 100 forecast times and report prediction failures
  separately from failed fits. Python 3.15.2, Julia and R agreed
  exactly on twelve Logistic forecasts at three held-out times.
  Independent Logistic predictions and percentile calculations matched
  every bound within 1e-10. The focused Rust regression and Clippy passed.

- [x] Verify multiseries bootstrap forecast intervals from the noisy
  joint Norton–Bass fit. Python 3.15.2, Julia and R agreed exactly
  for three refitted paths at time 15. The baseline matched both SciPy
  reference levels within 0.01. Replaying each parameter draw against
  the full covariate history and independently taking percentiles
  reproduced both series' interval bounds within 1e-9.

- [x] Carry forecast column identities into bootstrap intervals and
  require refitted forecasts to match the baseline column order.
  Python 3.15.2, Julia and R agreed on Norton–Bass series_1 and
  series_2 with identical values and intervals to the prior fixture.
  The focused Rust scalar regression and Clippy passed.

- [x] Verify the minimum bootstrap forecast sample count. Python
  3.15.2, Julia and R agreed on the noisy Logistic fixture with two
  residual draws. Every bound matched independent linear interpolation
  within 1e-10. A Python paired-case run with seed 42 had one successful
  refit and returned a null interval, as documented.

- [x] Add a native regression for a paired-case Logistic bootstrap
  with one failed fit and one successful forecast. The Rust test
  confirms a null interval and zero forecast failures.

- [x] Expose bootstrap_forecast_frame in Python Polars. The rebuilt
  Python 3.15.2 wheel retained series order and interval bounds from
  the saved native Norton–Bass response. The focused consumer test
  verified nullable Float64 bounds and Ruff passed.

- [x] Qualify bootstrap forecast tables for Arrow interchange.
  Python 3.15.2 Polars round-tripped the native Norton–Bass table and
  the single-refit Logistic table. Julia Arrow and R Arrow preserved
  the two series' order and the Logistic null Float64 bounds.
