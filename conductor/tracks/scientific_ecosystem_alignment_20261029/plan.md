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

- [x] Reconcile the remaining covariate/event, substitution and
  bootstrap ledger work with the discrete timing and mean-path
  interval evidence. Development validation passed. Release still
  fails closed with all 38 required obligations, including these three.

- [x] Expose native execution-mode discovery with the shared local CPU
  worker limit and explicit unqualified accelerator, distributed and
  scheduler modes. Python 3.15.2, Julia and R agreed. Two Rust batch
  regressions and Clippy passed. The installed sbatch and qsub wrappers
  could not run their version probes because Docker was unavailable.
  Scheduler and accelerator qualification remain open.

- [x] Let the native benchmark runner honor a Slurm task CPU budget.
  A local test supplied SLURM_CPUS_PER_TASK=2 and completed all three
  fast cases, as did the serial run. A zero budget was rejected before
  writing output. The Rust benchmark policy test and Ruff passed.
  Scheduler-backed runtime evidence remains open.

- [x] Record the requested worker budget, its source and the requested
  policy in benchmark receipts. A local Slurm environment fixture
  capped 64 requested CPUs at 32 workers, and a CLI run recorded two.
  Both completed all three fast cases. Ruff passed. These receipts
  record the worker request and source, with scheduler runtime still open.

- [x] Assert exact local benchmark parity across worker budgets.
  The three fast cases returned identical ordered results at requested
  budgets of 1, 2 and 32. The focused Rust test and Clippy passed.
  Scheduler-backed traces and accelerator evidence remain open.

- [x] Verify scheduled benchmark parity on a rebuilt Python 3.15.2
  wheel. Lotka–Volterra competition and lock-in both passed at
  requested worker budgets of one and two. The complete native
  results and order matched exactly. This was a local execution.

- [x] Verify the scheduled benchmark policy across Python 3.15.2,
  Julia and R. With two requested local workers, Lotka–Volterra
  competition and lock-in completed in the same order and produced
  identical native results across the three bindings.

- [x] Record elapsed kernel-call time and runtime platform in native
  benchmark receipts. A local Python 3.15.2 Darwin arm64 run completed
  all three fast cases and recorded a positive duration. Ruff passed.
  Scheduler traces, memory measurements and accelerator evidence remain open.

- [x] Record benchmark process peak resident memory in bytes on
  Darwin and Linux. The local Python 3.15.2 Darwin arm64 run passed
  all three fast cases and reported 65,863,680 bytes. This is the
  runner process high-water mark, including imports. Ruff passed.
  Linux runtime and scheduler-backed measurements remain open.

- [x] Verify benchmark receipt memory unit conversion with injected
  process counters. The runner converted 4,096 Linux KiB to 4,194,304
  bytes, kept the same Darwin byte count, and emitted null for the
  unsupported Windows counter. Each fixture ran the native fast suite.
  Lima reported no local guest. Actual Linux and scheduler runs remain open.

- [x] Capture local scheduled workload timing and process RSS receipts
  on Darwin arm64 with Python 3.15.2. One and two requested workers
  returned identical native results for both cases. The single runs
  recorded 14.06 s and 66,322,432 bytes, and 9.25 s and 66,895,872 bytes,
  respectively. These are individual local observations. Scheduler and
  accelerator comparisons remain open.

- [x] Run the full locked Rust workspace suite at `c68ac300`. All 254
  tests passed across the core, FFI and Polars crates. This verifies
  the local workspace suite. The release acceptance gates remain open.

- [x] Run the full Python native consumer test directory against a
  freshly built locked wheel at `0941cfe1`. Python 3.15.2 passed all
  83 tests, with repository default pytest options cleared for the
  focused directory run. Coverage and release gates remain separate.

- [x] Run the Julia binding suite and freshly installed R native
  test script at `e86679c5` against the rebuilt locked Rust FFI library.
  Both suites exited successfully. Package release gates remain open.

- [x] Repair the R source-package check. Bundle byte-identical copies
  of the seven Rust reference fixtures used by the R tests, document
  the exported helpers, and retain Apache-2.0 in installed licenses.
  R 4.3.0 on Darwin arm64 built the tarball with vignettes and
  R CMD check --no-manual returned Status: OK against the locked
  external Rust FFI library. Distribution of that library remains separate.

- [x] Verify local bundled R library discovery. Stage the locked
  Darwin arm64 debug dylib under inst/lib, build a binary package,
  install its archive into a second library, and unset the environment
  override. R 4.3.0 resolved the bundled dylib and returned the expected
  Logistic forecast. Release builds and other platforms remain open.

- [x] Verify local Julia bundled-library discovery. Julia 1.13.3
  developed the staged package in an empty project with offline
  dependency resolution. With no environment override it loaded
  the locked Darwin arm64 debug dylib from deps and returned the
  expected Logistic forecast. Artifact distribution remains open.

- [x] Build a Python source archive at `569b33ff`, extract it outside
  the checkout root, and build a locked CPython 3.15 Darwin arm64
  debug wheel from the extracted source using the existing Cargo
  target cache. An isolated wheel extraction returned the expected
  Logistic forecast through Polars on Python 3.15.2. Twine and
  check-wheel-contents passed. Uncached and release builds remain open.

- [x] Build the Python release wheel from the source archive at
  b29505d4 with an empty Cargo target directory and locked dependencies.
  The CPython 3.15 Darwin arm64 wheel passed Twine and
  check-wheel-contents, and an isolated Python 3.15.2 import returned
  the expected Logistic forecast through Polars. Wheel SHA-256
  a0faf2951863e87545ee0a6bc08cfa903cf8317733c9e6397c1038b223caa635. Other platforms and broader release acceptance remain open.

- [x] Run all 83 Python native consumer tests against the Darwin arm64
  release wheel built from the b29505d4 source archive. Python 3.15.2
  passed the full native test directory in 10.11 seconds with default
  pytest options cleared for this run. Coverage remains separate.

- [x] Verify bundled Julia and R consumers with the locked Darwin arm64
  optimized Rust FFI library at 95e243d9. Julia 1.13.3 loaded the library
  from the staged package in a fresh project. R 4.3.0 loaded it from
  a binary archive installed into a second library. Both produced the
  expected Logistic forecast with the environment override unset.
  Shared library SHA-256 e81de7d01b8f8a56095313efcfae1e254df47c4cd302c3b90207612722b84876. Other platforms and distribution gates remain open.

- [x] Run the complete Julia binding suite and R native test script
  using the staged and installed packages with the verified Darwin
  arm64 release library and no environment override. Both passed.
  This extends the bundled smoke check to the checked-in binding suites.

- [x] Reconcile local Darwin arm64 release artifact provenance.
  The Python archive revision b29505d4 and the Julia/R build revision
  95e243d9 differ only in this plan. The installed R and staged Julia
  libraries match the built FFI dylib byte for byte. R binary archive
  SHA-256 61fc91c8da5424edb4031574789411d971cbb247b1e96d33f56647cf40c21caf.
  Development validation passed. The release gate still reports all
  38 open obligations, and hosted distribution evidence remains open.

- [x] Measure Python binding coverage against the verified Darwin
  arm64 release wheel. All 83 native consumer tests passed on
  Python 3.15.2. The binding covered 160 of 199 statements
  (80.40%), clearing the explicit 80% threshold. Rust coverage and
  broader release evidence remain separate.

- [x] Measure locked Rust workspace line coverage at 57cb0ff9. The
  instrumented suite passed all 254 tests. Report generation reused
  the collected profiles with Homebrew LLVM 23.1.4 after the default
  toolchain lacked llvm-profdata. Workspace line coverage was 87.55%
  (15,403 of 17,594), including the unexercised PyO3 crate. Installed
  binding coverage and release acceptance remain separate.

- [x] Audit locked runtime dependencies at c6aa51f5. PyPI audit
  reported no known vulnerabilities for Polars and its runtime
  package, both 2.1.0. Cargo audit reported zero vulnerabilities
  against RustSec database 309392b18f6b17022f82c701f300433126901068
  and retained the bincode unmaintained warning RUSTSEC-2025-0141.
  Cargo tree showed no reverse dependency in the all-features,
  all-targets workspace graph. Broader supply-chain gates remain open.

- [x] Trace the bincode warning to pinned Polars. Cargo metadata
  retains bincode 2.0.1 under the optional polars-utils serde feature.
  The Darwin tabular feature tree omitted that edge, and the fresh
  release build log and release target directory contained no bincode
  compilation or artifacts. The lockfile audit warning remains open.

- [x] Compare repeat Python release archives on Darwin arm64. The
  source tarballs were byte-identical. Repackaging the same compiled
  inputs preserved every wheel payload except the CycloneDX SBOM and
  its RECORD entry. The SBOM differed only in timestamp and UUID.
  Byte-identical wheels and independent build reproducibility remain open.

- [x] Repackage the Darwin arm64 Python release wheel twice with
  SOURCE_DATE_EPOCH set to the source revision commit time. The two
  wheels were byte-identical and retained the CycloneDX SBOM with a
  fixed timestamp and no generated UUID. Twine and wheel-content
  checks passed. SHA-256 b2b4bbb9881b8ce3086be5abaedab2dfc4633fbe14d737758a0372fa39647852.
  This comparison shared the Cargo target directory. Independent
  build reproducibility remains open.

- [x] Compare an independent fixed-epoch Darwin arm64 wheel build
  from a fresh extraction and empty Cargo target. The compiled
  extension and all wheel members matched except the CycloneDX SBOM
  and its RECORD entry. Substituting the extraction root in ten
  SBOM references yielded identical JSON. Build-path-independent
  archive bytes and publication qualification remain open.

- [x] Normalize source-root URIs in unsigned wheel SBOMs while
  preserving the SBOM and updating RECORD. The two independently
  built fixed-epoch Darwin arm64 wheels became byte-identical.
  All RECORD hashes and sizes verified, the local build path was
  absent, and Twine, wheel-content, Ruff and a Polars forecast passed.
  Hosted packaging and publication integration remain open.

- [x] Add wheel SBOM normalization regressions for differing source
  roots, deterministic output, refreshed RECORD digests and signed
  input rejection. Both Python 3.15.2 tests and Ruff passed.

- [x] Wire fixed-epoch source-archive builds and unsigned wheel SBOM
  normalization into the PyPI and TestPyPI workflow definitions.
  Build with locked inputs, normalize before package validation, and
  check wheel contents before publishing. Actionlint and both normalizer
  regressions passed. Hosted execution remains unverified and the
  scientific release gate still precedes packaging.

- [x] Share the release wheel build between both publication workflows
  and the scheduled/manual release-readiness lane. The read-only lane
  builds and validates packages independently of the scientific gate.
  Publication retains the scientific gate and all three paths pin the
  Rust toolchain. Actionlint, ShellCheck and the normalizer tests passed.
  Hosted execution remains open.

- [x] Run the shared release build script end to end on Darwin arm64
  with the repo-local Python 3.15.2 environment and separate output
  directory. Source archive, locked release wheel and SBOM normalization
  completed. Twine, wheel-content checks and a Polars forecast passed.
  The run reused a Cargo target. Hosted execution remains open.

- [x] Configure the read-only release-readiness lane to retain
  validated source archives, normalized wheels and SHA-256 sums
  under the tested commit ID. Actionlint passed. Hosted execution
  and artifact readback remain open.

## Kairos resource peaks and historical binding regression

- [x] Return peak queue depth and peak concurrent service after resource
  dispatch. Immediate admission contributes only to concurrent service.
  The horizon and replay fixtures also verify the existing slot-time and
  wait-time totals.
- [x] Verify both peaks on the same seeded FIFO case in the rebuilt Python
  3.15.2 wheel, Julia and freshly installed R. The fixture reaches two
  waiting agents and one agent in service. All 27 network and 74 native
  boundary tests passed. Formatting and focused Clippy passed.
- [x] Recheck existing historical and forcing paths on the rebuilt core.
  The selected Python slice passed 13 tests. Julia and installed R passed
  their native conformance suites.

The broader historical-model, ABM and migration obligations remain open.
The packaging workflow has successful hosted evidence at earlier revisions.
Hosted release readiness for this worktree revision remains open.

## Kairos hazard numerical range

- [x] Preserve finite unweighted imitation hazards when the intermediate
  exposed-neighbour product overflows. Apply the exposure fraction first
  in this case and retain the arithmetic of existing finite unforced paths.
  The regression failed before the fix with an invalid hazard error.
  All 28 Rust network tests, formatting and focused Clippy passed.
- [x] Verify the same three-agent case in the rebuilt Python 3.15.2 wheel,
  Julia and freshly installed R. All returned counts 2 and 3, with agent 2
  completing service at tick 2. Broader migration gates remain open.

- [x] Qualify heterogeneous imitation when the effective exposure keeps
  a weighted or unweighted agent hazard finite.
  The half-exposed agent has coefficient 1e308 and multiplier two.
  The weighted and unweighted Rust regressions preserve the finite
  effective rate. All 29 network tests, formatting and focused Clippy
  passed. The weighted fixture reproduced adoption counts 1 and 2
  and completion at tick 2 in the rebuilt Python 3.15.2 wheel,
  Julia and freshly installed R.

## Near-unit-root SARIMA qualification

- [x] Compare four 240-observation AR(1) and MA(1) fits generated with
  coefficients +0.985 and -0.985 against independent R maximum
  likelihood. The installed R binding's Rust coefficients agreed
  within 8e-6 and likelihoods within 8e-7. The negative MA estimate
  stayed invertible at -0.999998093. Reproduce with
  `evidence/sarima-reference/check-near-unit-root.R`.
  These cases extend the optimizer evidence to both signs near the
  stationarity and invertibility boundaries. Higher-order fits near
  those boundaries and cross-binding validation of these cases remain open.

- [x] Verify the four near-unit-root cases through the installed Python
  3.15.2 wheel and Julia public fitting helpers, using the same samples
  and fitted-value reference exported from the installed R binding.
  All 240 fitted positions per case agreed within 1e-10, with identical
  null or missing positions. The wheel and FFI library were built at
  63248eab. The intervening changes are Conductor evidence only.

- [x] Qualify second-order AR and MA fits near positive and negative
  unit roots against R maximum likelihood. The positive-root MA(2)
  fixture exposed a coordinate-search stall along the invertibility
  boundary, with likelihood 0.193 below R. When an individual sweep
  stalls, test opposing moves in adjacent MA terms before reducing
  the step. The repaired fixture agrees within 6e-8 and the other
  three within 8e-7. Reproduce with
  `evidence/sarima-reference/check-second-order-boundary.R`.
  The six Rust SARIMA boundary tests, focused Clippy and the existing
  three higher-order R comparisons passed. Python and Julia checks
  of the new second-order cases remain open.

- [x] Verify both signs of the AR(2) and MA(2) boundary fixtures through
  the Python 3.15.2 wheel built from c0cb9f11 and the Julia fitting
  helper using the FFI library from the same source. All 300 fitted
  positions per case match the installed R binding within 1e-10,
  including agreement on missing positions.

## Nested Arrow IPC handoff to Julia

- [x] Qualify the Julia-side Rust IPC rewriter with a two-chunk
  Polars 2.1.0 file. Arrow.jl 2.8.1 read the rewritten file with the
  same column types and values as the direct Polars stream. Verified
  Int32 IDs, nullable Float64 list elements, null and empty lists,
  and Unicode, missing and empty labels. The Arrow reader represents
  the ID column as Union{Missing, Int32} on both paths.
- [ ] Direct readback of the untouched Python Polars IPC file remains
  open. Arrow.jl 2.8.1 returned an empty column set for this fixture.

## CSV identifier fidelity

- [x] Read competition identifiers, policy event labels and network
  endpoints as strings at the Python Polars CSV boundary. The network
  regression previously collapsed 001 and 1 into a duplicate edge.
  The rebuilt Python 3.15.2 wheel passed all three selected ingestion
  tests and Ruff passed. Julia and installed R re-ingested the network
  artifact with its original identifiers and checksum.
- [x] Extend installed-wheel CSV ingestion regression coverage to
  competition IDs and policy labels. Across all three dataset kinds,
  numeric-looking strings retain their spelling through checksum
  re-ingestion, and a missing identifier fails the Rust identifier
  diagnostic. All three cases passed on Python 3.15.2.

## Adoption dataset domain validation

- [x] Reject negative adoption counts and rates at the shared Rust
  dataset boundary. The monotone negative fixture passed diagnostics
  before the fix and now reports nonnegative:adoption. All 75 native
  boundary tests and focused Clippy passed. Installed binding validation
  for this diagnostic remains open.
- [x] Verify count and rate failures through the Python 3.15.2 wheel,
  Julia and freshly installed R using native builds from 7d9677b1.
  Both fixtures returned nonnegative:adoption as the sole failed
  diagnostic in each binding's structured ingestion error.

## Workspace and binding regression checkpoint

- [x] At source 87011793, the locked Rust workspace passed (257 tests).
  The installed Python 3.15.2 wheel passed all 86 tests under tests/native,
  Julia passed 253 assertions, and installed R passed native.R. The
  binding builds are from 7d9677b1, which has identical native and binding
  source. The development migration policy passes and the release gate
  still rejects 38 open scientific and acceptance obligations.

## Scenario threshold interpolation precision

- [x] Preserve narrow floating-point brackets in linear crossing times.
  A threshold one representable value above 1.5 and one below the upper
  sample previously interpolated at two-thirds of the interval instead
  of its midpoint. Use direct differences when the span is finite and
  retain scaled arithmetic for overflowing spans. Both precision tests,
  nine scenario boundary tests, formatting and focused Clippy passed.
- [x] Verify the narrow bracket through the public scenario grid in the
  rebuilt Python 3.15.2 wheel, Julia and freshly installed R. A Logistic
  path with samples 1.5 and 1.5000000000000004 crossed the intermediate
  threshold at time 0.5 in every binding. Julia and R also confirmed
  the first sampled crossing remains time 1. R used raw request JSON
  to preserve the threshold token.

- [x] Preserve paired ensemble standard errors for nearly equal final
  effects by centering before scaling. The native regression resolves
  effects one representable value apart and retains a finite standard
  error for opposite effects at the floating-point limit. Nine scenario
  boundary tests, formatting and focused Clippy passed. This check
  covers the Rust core. Broader ensemble qualification remains open.
- [x] Verify the adjacent-effects fixture through the public scenario grid
  in the rebuilt Python 3.15.2 wheel, Julia and freshly installed R.
  Two Logistic pairs produced final effects 1.5 and 1.5000000000000002.
  Each binding returned standard error 1.1102230246251564e-16, within
  four machine epsilons relative to the analytic half-gap. R received
  raw request JSON to retain the adjacent floating-point tokens.

## Paired ensemble sign-flip inference

- [x] Add optional exact_sign_flip inference for two through sixteen
  pairs. Enumerate every sign assignment to final paired differences
  and report the two-sided tail proportion, including ties. Inference
  assumes independent pairs and exchangeable signs under the null.
  The public Rust scenario boundary returns p = 1 for effects 50 and
  -25 and p = 0.5 for effects 50 and 25. Focused Clippy passed.
  Installed Python, Julia and R validation remains open.
- [x] Verify both sign-flip fixtures through the rebuilt Python 3.15.2
  wheel, Julia and freshly installed R. All three returned p-values 1
  and 0.5 for the same two Logistic comparisons. Python also verified
  the four assignments and extreme counts of four and two.
- [x] Document the exact sign-flip comparison request, output,
  two-to-sixteen-pair limit and inference assumptions in the native
  migration guide. The example uses the two-pair case already checked
  in all three bindings.
- [x] Verify the sign-flip workload boundary in the public Rust API.
  Sixteen zero-effect pairs count all 65,536 assignments as ties
  and return p = 1. Seventeen pairs return invalid_comparison while
  retaining all scenario executions. The targeted test and Clippy passed.

- [x] Reject scenario comparisons that request threshold interpolation
  without a threshold. The Rust boundary returns a comparison error
  and retains the scenario executions. The targeted test and Clippy
  passed.

- [x] Allow counterfactual event schedule replacement on explicit
  prediction requests as well as fitted states. The Rust Lotka–Volterra
  fixture removes the intervention from both baseline representations
  and verifies identical contrasts against the analytic trajectories.
  The causal boundary test and focused Clippy passed. Installed
  binding checks for the explicit request remain open.
- [x] Verify the explicit Lotka–Volterra event override through the
  rebuilt Python 3.15.2 wheel, Julia and freshly installed R. All three
  reproduced the analytic baseline and event-free counterfactual for
  the first product within 1e-8. R used raw JSON for the empty schedule.
- [x] Allow counterfactual covariate history replacement on explicit
  prediction requests. The Lotka–Volterra fixture verifies identical
  results for explicit and saved-state forecasts against the analytic
  contrast. The causal boundary test and focused Clippy passed.
  Installed binding checks for explicit histories remain open.
- [x] Verify the explicit Lotka–Volterra covariate history override in
  the rebuilt Python 3.15.2 wheel, Julia and freshly installed R using
  native source 8436bf99. All three matched the analytic first-product
  baseline and zero-price-history counterfactual within 1e-8. R used
  raw JSON to preserve the single forecast observation and covariate name.
- [x] Allow a counterfactual to introduce parameter events when the
  baseline constructor options have no schedule. The Lotka–Volterra
  fixture verifies that adding the intervention reverses the baseline
  and alternative trajectories of the existing event-removal case.
  The Rust causal boundary test and focused Clippy passed.
- [x] Verify event introduction with an unscheduled Lotka–Volterra
  baseline in the rebuilt Python 3.15.2 wheel, Julia and freshly
  installed R at native source 078feb6c. All three match the analytic
  first-product baseline and intervention values within 1e-8.
- [x] Document the counterfactual prediction payload and override scopes.
  The published-guide JSON example matches the event-introduction
  payload verified in Python, Julia and R.
- [x] Return the shared time grid with scalar and multi-product
  counterfactual forecasts and reject mismatched row counts.
  The causal Rust boundary fixture checks both result shapes.
  Formatting and focused Clippy passed.
- [x] Verify the returned prediction grid through Python 3.15.2,
  Julia and freshly installed R using native source 188d6f3d.
  All three returned times 1 and 2 and two aligned rows for each
  Lotka–Volterra trajectory. Scalar grid coverage remains in Rust.
- [x] Reject unknown counterfactual forecast fields and conflicting
  model aliases at the Rust boundary. This prevents a misspelled
  intervention field from silently producing an unchanged trajectory.
  The causal boundary test, formatting and focused Clippy passed.
- [x] Verify a misspelled counterfactual parameter override through the
  rebuilt Python 3.15.2 wheel, Julia and freshly installed R at
  5c5c9a2b. All three surfaced the native unknown-field error.
  The conflicting-alias regression is verified at the Rust boundary.
- [x] Preserve representable percentages when the intermediate ratio
  underflows. The Rust unit fixture checks both signs of the smallest
  subnormal effect and an ordinary percentage at the maximum finite
  scale. The causal boundary test and focused Clippy passed.
- [x] Use compensated accumulation in the scaled causal outcome mean.
  The public Rust counterfactual fixture retains the analytic effect
  of one-third when large opposite observations surround a small
  observation. The causal boundary test and focused Clippy passed.
- [x] Verify the cancellation fixture through the rebuilt Python 3.15.2
  wheel, Julia and freshly installed R using native source 958843bc.
  All three public causal summary calls reproduced the analytic
  one-third effect within 1e-15.

## Local MPI launcher qualification

- [x] Launch the fast native benchmark policy in two independent
  processes using Open MPI 5.0.11 on Darwin arm64. Both ranks used
  one worker, Python 3.15.2 and the wheel from native source 958843bc.
  Each passed three cases and produced identical benchmark results.
  Kernel elapsed times were 9.17 and 9.18 ms, with process peak RSS
  of 66,060,288 and 66,289,664 bytes. This verifies rank-local
  execution and separate receipt output. Cross-rank computation,
  scheduler allocation and accelerator qualification remain open.
- [x] Add opt-in Open MPI benchmark sharding using native case discovery.
  Each rank executes its disjoint slice and records rank, size and case IDs
  in its receipt. Two local ranks covered all three fast cases exactly once,
  and their combined result rows matched the serial run. Python 3.15.2 and
  Ruff passed. Multi-node, scheduler and accelerator qualification remain open.
- [x] Expand a filename's {rank} token for MPI receipts and create
  each receipt exclusively. Two local Open MPI ranks using one
  output template wrote distinct files and passed all three fast
  cases. An attempt to reuse a receipt path failed and preserved
  the original bytes. Ruff passed on the runner.
- [x] Verify local two-rank MPI sharding of the scheduled policy.
  Lotka–Volterra and Lock-in ran once each, and the combined result
  rows exactly matched the serial policy. Kernel elapsed time by rank
  was 7.84 s and 4.00 s, with peak RSS of 66,568,192 and 66,338,816
  bytes on Darwin arm64. Runner source was 05d143e5 with the unchanged
  native wheel from 958843bc. The scheduled policy selected the cases
  locally. Actual scheduler allocation and multi-node tests remain open.
- [x] Document MPI case sharding, rank receipts and per-rank worker
  budgets in the native migration guide. The example ran locally
  under Open MPI and produced both expected receipts.
- [x] Qualify local two-rank sharding for the manual benchmark policy.
  The ranks passed eight and seven distinct cases, and all fifteen
  result rows matched serial execution exactly. The Python 3.15.2
  run used the native wheel from 958843bc and runner at c6bb97b7.
  Local sharding now covers all three policies. Hosted scheduler,
  multi-node and accelerator qualification remain open.
- [x] Reconcile the accelerator/HPC ledger with the verified native
  CPU benchmark sharding across all three policies. Record bounded
  native status and retain multi-node, scheduler, accelerator and
  release obligations. Development policy passes. The release check
  continues to block all 38 open obligations, including accelerator/HPC.
- [x] Include the compiled Rust kernel version in native benchmark
  response metadata, which the benchmark runner retains in each
  receipt. The policy boundary test and focused Clippy passed.
  Version metadata identifies the release version. Exact build
  provenance still requires the source revision and artifact digest.
- [x] Verify compiled kernel version readback in two local MPI rank
  receipts. The rebuilt Python 3.15.2 wheel from d6b7e3ba produced
  version 0.7.0-alpha.1 in both receipts and passed all three fast
  cases across disjoint shards. Wheel SHA-256
  f069eff573a4c489926ccb324cb7877109eb753b132dfbf60d1e89e0ed720d31.
- [x] Record SHA-256 of the imported Python native extension in each
  benchmark receipt. Both local MPI rank receipts matched an independent
  digest of the installed extension from d6b7e3ba and passed all three
  fast cases. Ruff passed. The digest identifies the extension binary.
- [x] Add a serial-versus-MPI receipt verifier and document its use.
  The checker accepted both local fast-policy shards in reversed input
  order and rejected a missing rank and a substituted extension digest.
  Ruff passed. Multi-node, scheduler and accelerator gates remain open.
- [x] Verify all fifteen manual benchmark cases against a serial receipt.
  Require a non-sharded reference with unique case IDs and integer
  rank metadata. A two-rank local run passed verification and the
  checker rejected substitution of a shard for the serial reference.
  Ruff passed. Multi-node and scheduler execution remain open.
- [x] Verify MPI policy completeness against discovery from the installed
  kernel whose extension digest matches the serial receipt. The local
  fast policy passed and the checker rejected a consistent truncation
  of both serial and rank receipts. Ruff passed.
- [x] Add opt-in Slurm task sharding based on step rank and size.
  The runner shares case selection and receipt protection with MPI,
  identifies the launcher, and accepts the existing per-task worker
  budget. Injected Slurm step variables covered all three fast cases
  with two tasks and matched the serial receipt. Ruff passed.
  The local host has no srun executable. Live Slurm qualification remains open.
- [x] Reject mixed launchers in benchmark receipt verification.
  Legacy MPI receipts without a launcher remain readable. A local
  two-rank MPI set passed and replacing its second rank with an
  equivalent simulated Slurm receipt failed. Ruff passed.
- [x] Document the Slurm task launch, shared receipt path and
  verification procedure. The documented runner options passed with
  simulated step variables and two receipts matching serial. Live
  Slurm execution remains open.
- [x] Record Slurm job and step IDs and require them to agree across
  receipts. Simulated matching tasks passed serial verification and
  a receipt from a different step was rejected. Ruff passed.
  Live scheduler qualification remains open.
- [x] Retain a native consumer regression for Slurm receipt validation.
  The runner and verifier accept reversed receipt order for one simulated
  step and reject a matching case from another step. The installed Python
  3.15.2 test and Ruff passed. No local Slurm executable or Lima guest
  was available for live allocation evidence.
- [x] Expose native benchmark catalog and ordered case selection in
  execution-mode discovery for external launchers. Rust discovery and
  policy boundary tests passed. Scheduler, accelerator and distributed
  kernel qualification remain open.
- [x] Verify explicit benchmark selection order with three workers and
  an unknown case between two valid cases selected in reverse catalog
  order. The native response retained the error in its input position
  and matched both successful rows to the policy run.
- [x] Verify explicit benchmark ordering and isolated unknown-case
  errors in the rebuilt Python 3.15.2 wheel, Julia and installed R.
  All three preserved the requested order and returned successes on
  either side of the failed case with three workers at 668f374f.
- [x] Add a scheduled and manual Ubuntu CI step for two local Open
  MPI ranks. Verify scheduled case results against the existing serial
  receipt and retain the rank receipts. Actionlint passed. Hosted run
  evidence, multi-node and live Slurm qualification remain open.
- [x] Rehearse the scheduled CI benchmark sequence on Darwin arm64.
  The two-worker serial run passed both scheduled cases. Two Open MPI
  ranks passed one case each, and the current verifier matched both
  receipts against serial. The wheel from 668f374f matches the native
  source at b99b35cf. Hosted Ubuntu execution remains open.
- [x] Retain the serial reference before Open MPI setup and record
  the launcher version with the rank receipts. Bound the hosted
  benchmark step to ten minutes and require its artifact to exist
  on the existing success/failure upload path. Actionlint passed.
  GitHub showed no run for b99b35cf and no remote integration branch.

## Local identifiability at extreme sensitivity scales

- [x] Scale finite sensitivity columns before squaring them for local
  rank diagnostics. Two independent columns with entries near 1e308
  now retain full rank. An unrepresentable raw norm is null in JSON
  while its normalized column remains usable. All eight Rust library
  tests and focused Clippy passed. Global identifiability remains open.
- [x] Regress the scaled-rank change at native source 9fcd4a1d across
  fitting quality, complementary goods, historical Logistic,
  multi-generation and competition. All 60 selected Rust tests and
  six competition and multi-generation tests in the rebuilt Python
  3.15.2 wheel passed. This is local fit regression evidence.
- [x] Verify the scaled-rank core in the complete Julia and freshly
  installed R native suites at 4d0beff1. Both passed, including
  the six-parameter Norton Bass rank assertion and extreme-domain
  fitting rejection. Core and binding source match 9fcd4a1d.
- [x] Check that extreme-scale normalization preserves genuine rank
  deficiency. Collinear columns at scales 1 and 1e308 contribute
  one direction, while an independent 1e-20 column stays below the
  existing sensitivity cutoff. Both Rust rank regressions passed.
- [x] Report the existing absolute sensitivity norm cutoff and the
  zero-based indices of columns it excludes from local rank.
  Extreme-scale rank regressions and focused Clippy passed.
- [x] Document the absolute sensitivity cutoff, zero-based excluded
  column indices and null raw norms in the migration guide. All four
  Rust fit-quality tests passed with the documented diagnostic fields.
- [x] Report an unavailable final gradient as null when residual and
  sensitivity products overflow and cancel to NaN. The optimizer no
  longer masks the unavailable component as zero in its report.
  The Rust regression, four fit-quality tests and focused Clippy passed.
- [x] Report the number of converged starts and the scaled residual
  loss gap to the best other finite start. A symmetric objective
  converges at two different parameters with gap zero. The Rust
  regression, four fit-quality tests and focused Clippy passed.
  A zero gap describes the tested objectives. Global uniqueness
  remains a separate scientific obligation.
- [x] Verify the multistart fields on an analytic Logistic fit in
  Python 3.15.2, Julia and freshly installed R at 32b007c0. Each
  converged-start count matched its per-start records and each
  alternative loss gap was nonnegative.
- [x] Expose final optimizer coordinates for each finite start.
  The symmetric Rust fixture now reports the distinct coordinates
  -1 and 1 alongside their identical objectives. The multistart
  regression, four fit-quality tests and focused Clippy passed.
  These are internal transformed coordinates. Global identifiability
  remains open.
- [x] Retain initial transformed coordinates for every attempted
  optimizer start, including starts rejected before iteration.
  The Rust fixture confirms that an invalid first start retains its
  coordinates and the successful second start keeps its original
  index. Both multistart tests and focused Clippy passed.
- [x] Retain initial scaled residual loss for every finite start.
  The analytic Rust regression distinguishes a start that improves
  from loss four to near zero from one initialized at the optimum.
  All three multistart tests and focused Clippy passed.
- [x] Preserve the prediction error on starts rejected before iteration.
  The multistart fixture verifies the reason and original index.
  An attempted fallback after invalid sensitivities failed the existing
  extreme-seed fit-quality gate, so the committed change is diagnostic
  only. Three multistart tests, four fit-quality tests and Clippy passed.
- [x] Document multistart report ordering, transformed coordinates,
  objective gaps and rejected-start errors. The three Rust fixtures
  covering those fields passed.
- [x] Verify selected-start coordinates and objective reporting on the
  public Rust fitting paths for Bass, Logistic, Gompertz and Fisher–Pry.
  The selected start matches the optimizer's final loss, its initial
  loss is at least as high, and both coordinate vectors match the
  fitted parameter count. All four fit-quality tests passed.
- [x] Verify per-start initial and final loss and coordinate vectors
  through the rebuilt Python 3.15.2 wheel, Julia and freshly installed
  R at 2a5d2c0e. All three matched the selected start to the optimizer
  summary on an analytic Logistic fit.
- [x] Qualify held-out Logistic forecasts from a perturbed training
  trajectory. The Rust fit converged with local rank three on 22
  training observations. Saved-state predictions at nine later
  times were within one adoption unit of the analytic trajectory.
  All five fit-quality tests and focused Clippy passed. This is
  point-forecast evidence for one deterministic perturbation fixture.
- [x] Verify the perturbed Logistic fit and saved-state holdout
  through Python 3.15.2, Julia and installed R. All three reported
  local rank three and nine predictions within one adoption unit.
  Reused native builds at 2a5d2c0e with identical runtime source.
- [x] Extend the deterministic held-out trajectory fixture to Gompertz.
  The Rust fit converged at local rank three and forecast nine later
  observations within one adoption unit of the analytic curve.
  All six fit-quality tests passed.
- [x] Verify the perturbed Gompertz fit and saved-state holdout through
  Python 3.15.2, Julia and installed R. All three converged with local
  rank three and forecast all nine held-out values within one adoption
  unit. Reused builds from 2a5d2c0e with identical runtime source.
- [x] Extend the deterministic holdout to Bass using its closed-form
  adoption curve. The fit converged with local rank three and nine
  held-out predictions within one adoption unit. All seven Rust
  fit-quality tests and formatting passed.
- [x] Verify the perturbed Bass fit and saved-state holdout through
  Python 3.15.2, Julia and installed R. All three converged with local
  rank three and forecast all nine held-out values within one adoption
  unit. Reused builds from 2a5d2c0e with identical runtime source.
- [x] Extend the deterministic holdout to Fisher–Pry using its
  analytic share curve. The Rust fit converged with local rank two
  and predicted all nine held-out shares within one percentage point.
  All eight fit-quality tests passed.
- [x] Verify the perturbed Fisher–Pry fit and saved-state holdout in
  Python 3.15.2, Julia and installed R. All three converged with local
  rank two and forecast all nine shares within one percentage point.
  Reused builds from 2a5d2c0e with identical runtime source.

## Bass interval coverage on a fixed observation grid

- [x] Run 64 paired trials on times 1 through 24 with p=0.03,
  q=0.21, m=120, Gaussian noise SD 0.5 and 100 bootstrap draws per
  trial and method. Nominal 90% case intervals covered 57/64
  (mean width 1.18044, zero failed refits). Centered residual
  intervals covered 55/64 (mean width 1.07634, 29/6400 failed
  refits). Both returned all 64 intervals without refusal.
  Reproduce with evidence/check-bass-bootstrap-coverage.py and the
  installed Python 3.15.2 wheel at 2a5d2c0e. This measures one
  synthetic design. Broader uncertainty calibration remains open.
- [x] Measure pointwise Bass mean-trajectory coverage at times 25, 30
  and 40 using the same 64 trials. Case resampling covered 55, 57
  and 57 trials. Centered residual resampling covered 56, 57 and 55.
  All 64 intervals were available at each time with no forecast failures.

## Kairos runtime boundary reconciliation

- [x] Cross-check the imported Kairos plan against the active Rust
  workspace at 86e00aed. The historical Python claim adapter and
  bindings/rust manifest named by I01–I02 are absent from this branch.
  Cargo resolves Kairos core, DES, ABM, RNG and types through
  innovate-core at d56863505d938feb069af88a737eb7668113d14b.
  Simulation and network code dispatch real scheduler or context
  events and use the pinned deterministic streams. The locked Kairos
  and network suites passed 10 and 29 tests, respectively.
  Use the root workspace and these runtime tests when mapping the
  imported acceptance criteria. Its remaining leaves retain their
  dependency and ownership gates.
- [x] Report actual Kairos dispatch counts in simulation metadata.
  Homogeneous DES and ABM count adoption dispatches. Network ABM
  counts dispatched requests and completions, including requests
  that subsequently balk. Initial adopters do not count as dispatches.
  The single-adoption network fixture reports two dispatched events.
  All 39 locked Kairos and network tests passed.
- [x] Verify network dispatch counts in the rebuilt Python 3.15.2 wheel,
  Julia and freshly installed R at f73933fe. The same seeded network
  reports two Kairos dispatches for one request and one completion
  in all three bindings.
- [x] Check homogeneous ABM dispatch counts against adoption gained
  after initialization. The existing partial and saturated population
  fixture verifies that initial adopters contribute zero dispatches.
  All ten locked Kairos tests passed.
- [x] Exercise equal-tick dispatch and cancellation directly through the
  pinned Kairos scheduler in the root workspace. Three tied events
  were scheduled, the middle one cancelled, and the remaining IDs,
  kinds and tick dispatched in insertion order before an empty step.
  All eleven locked Kairos tests passed. This supplies runtime evidence
  for the tie/cancel behavior in imported leaf I03.
- [x] Exercise pinned ABM entity and RNG replay in the root workspace.
  The fixture checks a missing component, dispatches a behavior to the
  spawned entity, updates its component with an entity-purpose draw,
  and verifies the same identity, value and RNG checkpoint on replay.
  All twelve locked Kairos tests passed. This supplies runtime evidence
  for the state and RNG portion of imported leaf I04.
- [x] Verify the root workspace split against imported I06. The locked
  innovate-core normal and build dependency tree with default features
  disabled contains no Arrow or Polars packages, and all 12 Kairos
  tests pass in that configuration. The separate innovate-tabular
  package passed all nine prediction and IPC tests. This verifies the
  package split on Darwin arm64.
- [x] Map the native Bass operation, schema, C ABI ownership and
  runtime evidence to imported I05. The contract names separate
  bridge and installed-consumer verification leaves. The locked
  innovate-ffi suite passed.
- [x] Verify the I05 C ABI bridge leaf. A seeded Bass DES request
  returned schema 2.0, counts zero and one, and one Kairos dispatch.
  An unsupported engine returned a structured error and null result.
  Both owned responses were released. Both locked FFI tests passed.
- [x] Verify I05 installed consumers with the same Bass DES fixture.
  Python 3.15.2, Julia and installed R returned counts [0, 1] and
  one dispatch, and all three rejected the unknown engine.
  Reused f73933fe artifacts with identical runtime source.
- [x] Derive the paired-outcome oracle for imported I08 from the
  documented diffusion_v1 fixture. Record ordered draw and time
  identity, strategy-specific parameter values, shared units and
  downstream perspective ownership. The fixture does not establish
  runtime exporter acceptance. Executable negative cases remain open.
- [x] Implement the bounded I08/I08b paired Bass validation boundary
  in the migrated core. A two-strategy, three-draw synthetic request
  round-trips unchanged. Missing or reordered draws, duplicate
  parameter keys, incompatible time units and perspectives fail.
  The public Rust boundary test and Clippy passed. Arrow export
  and installed consumer qualification remain open.
- [x] Exercise historical diffusion_v1 CSV rows through the rebuilt
  Python 3.15.2 wheel at a5c66d41. Polars grouped both strategies and
  their three draws in input order. Rust returned the grouped payload
  unchanged and rejected reversing the second strategy's draws.
  Fixture provenance and period units were supplied by the test.
  The targeted regression and Ruff passed.
- [x] Run the historical paired Bass payload through Julia and installed R
  against the local Rust FFI. Both returned the original JSON payload and
  rejected a reversed draw sequence with `invalid_request`. R comparison
  used parsed JSON equality, preserving array order and ignoring object key
  order. This verifies the JSON bridge. Arrow export remains open.
- [x] Encode validated paired Bass trajectories as Arrow IPC in the Rust
  tabular crate. Rows follow strategy, draw and time order, with nullable
  perspective and the original parameter arrays encoded as JSON. The
  Polars round-trip regression checks identity, adoption and parameter
  values and rejects reordered draws before export. All ten tabular tests
  passed. Host export adapters and downstream acceptance remain open.
- [x] Verify paired Bass Arrow export through a rebuilt Python 3.15.2
  wheel, Julia and a freshly installed R package on Darwin arm64.
  Polars decoded the same 24 historical rows from each binding, and
  all rejected a reversed draw sequence. R used JSON text to preserve
  the original numeric tokens. Downstream consumer acceptance is open.
- [x] Exercise downstream consumption of the historical paired Bass
  export. The installed Python wheel regression compares all 24
  trajectory keys and adoption values with the fixture and rejects
  reordered draws. Separately, Polars decoded the Julia IPC file and
  pivoted the terminal rows on draw ID. Voiage 3.1.0 computed EVPI of
  0.001 for three paired draws using a synthetic 0.021 alternative
  offset, matching the independent rational-arithmetic oracle to
  1e-12. This qualifies one consumer example. A production valuation
  adapter and supported-version matrix remain open.
- [x] Extend the near-unit-root SARIMA checker to seasonal MA at
  period four with generating coefficients ±0.95. The installed R
  binding agreed with independent maximum likelihood on both 600-row
  fixtures within 1e-5 in coefficient and 2.33e-7 in log likelihood.
  The four existing nonseasonal cases also passed.
- [x] Qualify simultaneous historical price replacement and event
  removal for a Lotka–Volterra counterfactual. The analytic fixture
  covers the event boundary, the final price knot and extrapolation.
  All 78 Rust boundary tests passed. Python 3.15.2, Julia and installed
  R matched the first product's analytic baseline and counterfactual
  within 1e-8 using the existing native artifacts.
- [x] Verify comparison failure isolation in a parallel scenario grid.
  A failed Logistic simulation invalidates its own comparison while
  forward and reverse comparisons retain order, rank and signed effects.
  Rust, Python 3.15.2, Julia and installed R passed the shared fixture.
  The Python Arrow readback retained the failure and both successes.
- [x] Allow paired ensemble comparisons at an explicitly selected
  sample on the shared grid. Report the evaluation time and retain
  final-sample evaluation by default. An off-grid time produces a
  comparison error while retaining all executions. The focused Rust
  regression verifies the selected effect against analytic Logistic
  output. Installed binding qualification for this option remains open.
- [x] Verify selected-time paired ensembles in the rebuilt Python
  3.15.2 wheel, Julia and installed R against native source 925da465.
  All reported time 2 by default, effects [50, 50] at selected time 1,
  and an invalid comparison at off-grid time 1.5 with all four
  executions preserved. Python Arrow readback preserved row status.
- [x] Document selected-time ensemble evaluation in the native guide.
  The installed Python wheel ran the documented paired example at time
  1 with bootstrap and sign-flip enabled. Effects were 50 and 25, the
  mean was 37.5, and the exact two-sided p-value was 0.5.
- [x] Read the paired Bass export directly with Arrow.jl 2.8.1 in an
  isolated offline consumer environment. Arrow.Table decoded all
  eight columns and 24 rows and retained the missing perspective
  values. Document the native file path in the Julia binding guide.
  Direct readback of files written by Python Polars remains separate.
- [x] Decode the R binding's paired Bass export using R Arrow 20.0.0.2.
  The checker matched all 24 historical scenario, draw, time and
  adoption rows, with missing perspectives and fixture provenance
  preserved. The macOS binary was built under R 4.3.3 and ran under
  R 4.3.0. Hosted consumer qualification remains open.
- [x] Qualify UInt64 maximum draw identity in paired Bass Arrow export.
  The Rust regression and installed Python, Julia and R exporters
  preserved 18446744073709551615. Julia and R used JSON text input.
  R Arrow 20.0.0.2 preserved the exact ID when cast to UTF-8 in the
  Arrow table. Direct data-frame conversion rounded it. Document the
  exact-ID read path in the R help.
- [x] Report over-capacity slot-ticks for Kairos network resources.
  Integrate active service above scheduled capacity through the
  horizon, including nonpreemptive service that continues after
  capacity falls to zero. The clipped fixture reports three excess
  slot-ticks and the retry replay reports zero. All 29 network tests
  and the focused replay test passed. Installed binding checks remain open.
- [x] Verify the over-capacity resource metric at native source
  1c4680cf in the rebuilt Python 3.15.2 wheel, Julia and installed R.
  The capacity drop at tick three gave three excess slot-ticks,
  five occupied slot-ticks and no completions by tick six in all
  three bindings.
- [x] Report unused configured Kairos capacity as idle slot-ticks.
  The capacity-drop fixture retains one idle slot-tick before closure
  even though total occupied service exceeds configured capacity.
  The no-excess retry fixture reports two idle slot-ticks. All 29
  network tests and the focused replay test passed. Installed binding
  checks for the new field remain open.
- [x] Verify idle slot-ticks at native source aad3a161 through a rebuilt
  Python 3.15.2 wheel, Julia and installed R. Each reported one idle
  slot-tick and verified that occupied plus idle equals configured
  plus over-capacity in the scheduled closure fixture.
- [x] Run the local regression gate at 0bf80a22. The locked Rust
  workspace passed 280 tests. The native Python suite passed against
  the installed 3.15.2 wheel, and the Julia and installed R native
  suites passed against the matching local shared library. Migration
  development policy passed with 38 open release obligations.
- [x] Retain idle and over-capacity assertions in the Python, Julia and
  R native suites that CI invokes. The fixed-capacity FIFO fixture
  reports five idle and zero excess slot-ticks in each binding.
  The affected sections passed locally with the installed artifacts.
- [x] Cover scheduled closure in the binding suites used by CI. At
  zero capacity after tick three, the active service completes at
  tick six. Each binding reports one idle slot-tick, three excess
  slot-ticks, and a single completion. The affected checks passed
  locally in Python, Julia and installed R.
- [x] Include the three external STL and SARIMA reference fixtures
  required by the Rust boundary tests in the source distribution.
  A fresh extraction built the locked test target in its own target
  directory and passed the selected SARIMA and STL tests.
- [x] Run all 79 native boundary tests from the clean source extraction
  and add the same locked check to scheduled and manually dispatched
  Linux CI. The local suite and actionlint passed. Hosted execution of
  the workflow remains open.
- [x] Include the Python native consumer tests and historical paired
  Bass CSV inputs in the source distribution. A fresh extraction
  passed all 81 packaged consumer tests against the installed Python
  3.15.2 wheel with matching runtime source. Repository policy and
  benchmark receipt tests continue to run from the checkout.
- [x] Build and install a Python 3.15.2 wheel from the extracted
  source distribution and pass all 81 packaged consumer tests.
  The local wheel build reused Cargo artifacts. Extend scheduled and
  manually dispatched Linux CI to build and test from the extraction.
  Actionlint passed locally. Hosted execution remains open.
- [x] Require a Boolean release readiness declaration and reject
  premature readiness during development validation. Eight policy
  regressions passed. The optimized Python release check still
  rejects the current ledger with 38 outstanding obligations.
- [x] Reject negative adoption values at the shared paired Bass
  validation boundary before JSON or Arrow handoff. The native
  contract and Arrow round-trip regressions passed.
- [x] Exercise negative adoption through the public Rust Arrow exporter.
  The regression requires the domain error before IPC bytes are returned,
  alongside the full-width identity round trip and draw-order rejection.
- [x] Share the Rust paired Bass validator directly with Arrow export.
  The exporter validates the in-memory payload and writes from it
  without serializing a kernel request or reparsing a response.
  The core contract and all ten tabular tests passed.
- [x] Verify direct Rust validation at a495bde5 through the rebuilt
  Python 3.15.2 wheel, Julia and installed R. All three produced
  equal 24-row Arrow tables and rejected the negative adoption fixture.
- [x] Require named JSON objects for paired strategies, draws and
  parameters. Serde can decode positional arrays as structs, which
  could reach the Arrow writer without named fields. The exporter
  now rejects the positional draw with a validation error. The
  focused Rust contract and IPC tests passed.
- [x] Keep malformed positional draw rejection in the packaged Python
  consumer regression. The rebuilt wheel returned the validation error
  and passed the historical fixture round trip and reordered-draw checks.
- [x] Add paired Arrow validation to the Julia and R native suites.
  Both returned an Arrow file for valid named draws and rejected
  positional draws through the rebuilt shared library. The affected
  sections passed locally.
- [x] Allow Kairos network resources to start closed. A seeded
  three-agent queue waits until two slots open at tick seven, then
  completes two services at tick nine and the final service at tick
  eleven. Negative and out-of-range capacities remain invalid.
  All 29 network tests passed. Installed-binding checks remain open.
- [x] Verify initially closed Kairos resources at 3f05ef36 with the
  rebuilt Python 3.15.2 wheel, Julia and installed R. All three
  reported counts [0, 0, 0, 2, 3] and completions at ticks 9, 9 and 11.
- [x] Document initially closed Kairos resources in the native guide.
  The installed Python wheel ran the guide's JSON options and matched
  the documented completion ticks 9, 9 and 11.
- [x] Verify initially closed Kairos resources through a scenario
  specification and its generated replay request. The replay retains
  explicit zero capacity and reproduces the complete simulation result
  after opening two slots. The focused Rust boundary test passed.
- [x] Add opt-in resource differences to paired Kairos scenario
  comparisons. Signed differences use alternative minus baseline for
  capacity, service and waiting totals. Comparisons require aligned
  tick resolution and horizon. The Rust fixture checks a delayed
  opening against an initially open resource and rejects mismatched
  tick resolution. Installed-binding checks remain open.
- [x] Verify resource comparisons at a2061a52 through the rebuilt
  Python 3.15.2 wheel, Julia and installed R. All reported a signed
  configured-capacity difference of minus three slot-ticks and
  rejected unequal tick resolutions while retaining all executions.
  Python Arrow readback preserved both comparison statuses.
- [x] Document the opt-in Kairos resource comparison and signed units
  in the native guide. The documented pair ran through the installed
  wheel and returned the verified minus-three configured slot-ticks.
- [x] Retain a Python Arrow regression for signed Kairos comparison
  totals, horizon and tick resolution. The same export retains the
  incompatible pair's error and null result. The focused installed
  wheel test and Ruff passed.
- [x] Verify the shared resource comparison fixture at the Rust request
  boundary. A successful Logistic execution can be compared by adoption
  but cannot supply resource totals. The resource comparison fails
  independently while the ordinary comparison succeeds. The focused
  boundary regression passed.
- [x] Verify the Kairos comparison fixture ships in the source archive.
  Both focused Rust and Python tests passed from a fresh extraction,
  using the existing Cargo target directory and matching installed wheel.
- [x] Add signed resource comparison checks to the Julia and R native
  suites. Julia reads the shared Rust fixture. R ships an identical
  fixture under its package tests. Both focused checks preserved the
  negative capacity difference and rejected the incompatible pair.
- [x] Expose Kairos resource comparison discovery on the network
  scenario family. Discovery identifies the option, engine, direction
  and alignment requirements. The focused Rust registry test passed.
- [x] Verify keyed Kairos resource discovery in the rebuilt Python
  3.15.2 wheel, Julia and installed R at 6709dd48. All expose the
  opt-in flag, signed direction and three alignment requirements.
  All ten scenario-filtered Rust boundary tests passed.
- [x] Allow custom network specifications with zero average degree
  and an empty explicit edge list. This lets scenarios isolate
  resource effects from imitation. The initially closed scenario
  replays with zero degree, and all ten scenario-filtered Rust
  boundary tests passed. Installed-binding verification remains open.
- [x] Verify zero-degree custom network specification and replay at
  28d0278a through the rebuilt Python 3.15.2 wheel, Julia and installed
  R. All returned counts [0, 0, 0, 2, 3] and reproduced the complete
  simulation result from the generated replay request.
- [x] Verify degree validation around the edgeless custom scenario.
  The Rust boundary rejects a zero-degree declaration with an explicit
  edge, a negative degree and a zero-degree ring. The valid custom
  graph still executes and replays.
- [x] Reject network specifications whose average degree exceeds the
  simple graph bound of node count minus one. This applies before
  execution or artifact import. All ten scenario-filtered Rust
  boundary tests passed, including a two-node degree-two rejection.
- [x] Exercise network degree bounds during legacy scenario artifact
  import. The focused Rust test preserves a zero-degree custom artifact
  without inventing replay inputs and rejects degree two on two nodes.
- [x] Reject blank time units in scenario specifications and artifact
  imports so aligned comparisons cannot declare a blank common basis.
  The legacy artifact regression and all ten scenario-filtered Rust
  boundary tests passed.
- [x] Enforce the declared horizon when applying a network scenario
  specification. Observations beyond it are rejected, including
  requests using the time alias. Observations at the horizon still
  execute and replay. All ten scenario-filtered boundary tests passed.
- [x] Apply the declared network horizon to resolved scenario requests
  and artifact replay validation. A replay with observations beyond
  its declared horizon fails both direct execution and import. All
  ten scenario-filtered boundary tests passed.
- [x] Verify declared network horizon errors at 3b57be07 through
  the rebuilt Python 3.15.2 wheel, Julia and installed R. Each rejects
  an observation at time 11 when the declared horizon is 10.
- [x] Verify that sparse observations preserve resource completions
  and totals across scheduled closure and reopening at a fixed horizon.
  All 29 network and 12 Kairos tests passed. The public kernel requires
  a strictly increasing observation grid.
- [x] Include peak queue and concurrent service differences in Kairos
  resource comparisons. These differences are agent counts. The
  focused Rust regression verifies one additional concurrent service
  for the two-slot reopening alternative.
- [x] Retain the peak concurrent service difference in the binding
  regression suites. The rebuilt Python 3.15.2 wheel, Julia and installed
  R all reported one additional concurrent service for the alternative.
  Python verified the value after Arrow export.
- [x] Include completed services, balked requests and abandoned
  requests as signed event counts in Kairos resource comparisons.
  The focused Rust test verifies three fewer completed services when
  a resource stays closed throughout the scenario.
- [x] Verify signed Kairos resource event counts at d87ee6ce in the
  rebuilt Python 3.15.2 wheel, Julia and installed R. The permanently
  closed alternative reports three fewer completions and no difference
  in balks or abandonments. Python also verifies Arrow readback.
- [x] Distinguish balks from abandonments in the Rust resource
  comparison regression. A closed resource with one queue place
  produces two extra balks, one extra abandonment and five extra
  abandoned wait ticks compared with the open baseline.
- [x] Verify resource comparison counts repeated requests from a
  retried agent. The closed no-queue alternative produced six extra
  balk events across three agents with one retry each and no extra
  abandonments. The focused Rust regression passed.
- [x] Compare agents waiting and in service at the shared horizon.
  The focused Rust scenario test ends at reopening and verifies
  one additional waiter and two additional agents in service.
- [x] Verify resource counts at the horizon through the rebuilt Python
  3.15.2 wheel, Julia and installed R at 50e49a55. All report one
  additional waiter and two additional agents in service. The Python
  check reads the comparison back from Arrow.
- [x] Retain horizon queue and service counts in the packaged Python
  Arrow regression. At reopening the exported comparison preserves
  one additional waiter and two additional agents in service.
- [x] Retain horizon resource count assertions in Julia and R native
  tests. Both shorten the existing fixture to tick seven and verify
  one additional waiter and two additional agents in service.
- [x] Qualify local policy timing inputs without event labels in CSV,
  Parquet and Arrow IPC. All three generate stable event IDs, retain
  signed effects and reproduce the checksummed artifact on re-ingestion.
  The three packaged Python regression cases passed.
- [x] Verify explicit local format selection on a file with a neutral
  extension. Polars reads the Parquet input and ingestion appends the
  load step while preserving upstream Unicode metadata and the caller's
  provenance object. The focused Python ingestion regressions passed.
- [x] Reject positional provenance arrays before dataset checksum
  generation. The native validation and checksum operations now
  require named provenance fields. Existing object encoding is
  preserved. All five dataset-filtered Rust boundary tests passed.
- [x] Verify that the rebuilt Python wheel raises NativeError with
  invalid_request for positional provenance in dataset checksum.
  The focused packaged regression and Ruff passed.
- [x] Verify malformed provenance checksum errors through Julia and
  installed R with the rebuilt Rust library. Both returned
  invalid_request with the named-object requirement.
- [x] Distinguish malformed provenance digests from content checksum
  mismatches. A supplied nonempty digest must be 64 hexadecimal
  characters. Uppercase SHA-256 remains valid. All five focused Rust
  dataset tests passed, including timestamp and tamper behavior.
- [x] Reject malformed supplied dataset digests before sorting,
  serializing or hashing content. The five focused Rust dataset
  tests passed, including valid uppercase digest and tamper checks.
- [x] Verify historical checksum encoding through dataset ingestion.
  A Unicode provenance fixture distinguishes the legacy Python digest
  from the Rust default. Re-ingestion reproduces the legacy artifact
  with the recorded encoding and rejects the mismatched default.
  All five focused Rust dataset tests passed.
- [x] Document re-ingestion with the artifact's recorded checksum
  encoding. The guide example reproduced a Unicode artifact using
  python_sorted_json_v1 through the installed Python wheel.
- [x] Retain historical checksum re-ingestion in the packaged Python
  suite. The Unicode fixture reproduces its artifact with the recorded
  Python encoding and rejects a checksum mismatch under the default.
  The focused test and Ruff passed using the existing native wheel.
- [x] Verify historical checksum re-ingestion through Julia and
  installed R against native source 7d9393ef. Both reproduced the
  Unicode artifact with its recorded encoding and rejected the
  mismatch under the Rust default.
- [x] Accept a complete dataset artifact directly in ingest_dataset.
  Rust uses the recorded encoding, verifies the required checksum and
  recomputes diagnostics. The five focused Rust dataset tests passed,
  including historical round-trip, tampering and conflicting inputs.
  Installed-binding checks remain open.
- [x] Verify direct dataset artifact re-ingestion at f0ddba9f in
  Python 3.15.2, Julia and installed R. All reproduced the legacy
  artifact and rejected modified adoption data. Python imported
  the rebuilt wheel from an isolated extraction after disk exhaustion.
- [x] Document direct dataset artifact re-ingestion. The guide example
  reproduced the historical Unicode artifact with the rebuilt Python
  wheel extracted for the binding check.
- [x] Retain direct artifact re-ingestion and tamper detection in the
  packaged Python regression. The rebuilt wheel extraction passed.
- [x] Retain direct artifact integrity checks in Julia and R native
  tests. Both reproduced the checksummed artifact and rejected a
  changed adoption value. The affected binding checks passed locally.
- [x] Verify artifact re-ingestion replaces a stale validation report
  with freshly computed diagnostics. The focused Rust test passed.
- [x] Verify artifacts directly from borrowed dataset and encoding fields
  without cloning a synthetic ingestion request. Copy the returned
  dataset only after verification and diagnostics. All five focused
  Rust dataset boundary tests passed.
- [x] Compare historical checksum encoding with CPython over a seeded
  sample of 2,047 finite binary64 values across the exponent range.
  The installed wheel digest matched the independent JSON/SHA-256 oracle.
- [x] Extend the historical checksum oracle with signed and unsigned
  integer extremes and nested Unicode keys spanning the BMP and
  supplementary planes. The packaged Python regression passed.
- [x] Validate reversed duplicate edges when directed is explicitly false.
  Directed datasets retain reciprocal edges, and legacy datasets without
  the flag retain ordered keys. The focused Rust edge test passed.
- [x] Verify Polars ingestion preserves directed reciprocal edges and
  rejects undirected duplicates with structured diagnostics. The
  focused Python test passed against the rebuilt native extension.
  Wheel assembly hit the internal disk limit, so the check used an
  isolated import directory on the repository volume.
- [x] Verify reciprocal edge diagnostics through Julia and installed R.
  Both accept explicitly directed reciprocal edges and identify the
  undirected duplicate with duplicate_keys:source_target.
- [x] Flag self-loops in network edge diagnostics as incompatible
  with Kairos network simulation. Generic datasets retain their
  self-loop data. The focused Rust edge regression passed.
- [x] Verify ingestion retains self-loop edges and regenerates their
  Kairos compatibility warning when re-ingesting the checksummed
  artifact. The focused Rust edge test passed.
- [x] Allow aligned empty edge columns for edgeless graphs. The Rust
  ingestion regression verifies checksum replay and rejects unequal
  column lengths. Node identity still comes from the scenario.
- [x] Verify edgeless dataset ingestion and artifact replay through Julia
  and installed R against the rebuilt Rust library.
- [x] Verify the edgeless Polars path with a rebuilt Python 3.15.2 binding.
  Staging on the repository volume allowed wheel assembly. The focused
  regression imported the rebuilt extension and verified artifact replay.
- [x] Verify header-only local CSV ingestion for edgeless networks.
  Both the CSV and in-memory Polars cases preserve empty edge columns
  and reproduce the artifact on replay.
- [x] Remove the legacy pytest.ini that shadowed the native settings
  in pyproject.toml. The checkout now uses the same configuration
  as the source archive. Thirteen focused Python ingestion checks and
  seven Rust dataset checks passed.
- [x] Run the full Python native suite via default discovery after
  pytest configuration consolidation. All 101 tests passed under
  Python 3.15.2 using the rebuilt extension in an isolated import
  directory, including policy and benchmark receipt tests.
- [x] Run the locked Rust native boundary suite at dcd670c1 after
  scenario and ingestion changes. All 81 tests passed, including
  artifact replay, resource comparisons and edgeless ingestion.
- [x] Run the complete Julia and installed R native suites at 4bb81f78.
  Both passed with the shared library rebuilt from the current Rust
  source. Temporary files were staged on the repository volume.
- [x] Generate optional policy labels and unit edge weights in Rust
  before checksumming. The Python Polars adapter supplies explicit
  columns when present. Seven Rust dataset checks and six focused
  Python cases passed with the rebuilt extension. Julia and R checks
  for the new defaults remain open.
- [x] Verify Rust-generated policy labels and edge weights through
  Julia and installed R. Both reproduced the normalized artifacts.
  R used raw JSON for the edge replay to preserve the 1.0 token.
- [x] Verify that omitted and explicit policy labels or unit edge
  weights yield identical artifacts and checksums in Rust.
- [x] Keep artifact verification byte-sensitive to removed defaulted
  fields. Ingestion fills omitted fields for bare datasets only.
  Removing labels or weights from a checksummed artifact now fails
  verification. All eight focused Rust dataset tests passed.
- [x] Verify checksummed datasets supplied without an artifact envelope
  also reject removed labels or weights. Defaults apply only when the
  incoming dataset supplies no checksum. The focused Rust test passed.
- [x] Document default normalization before checksumming and strict
  verification of checksummed inputs. All eight focused dataset tests
  passed against the documented behavior.
- [x] Reuse the normalized dataset when stamping an ingestion artifact.
  Inputs with no defaults stay borrowed through verification. All
  eight focused Rust dataset tests passed.
- [x] Report a validly labelled self-loop even when another edge has
  invalid endpoints. The focused Rust test verifies both diagnostics.
- [x] Report duplicate edge keys only for actual duplicate valid pairs.
  An invalid endpoint is diagnosed independently and does not create
  a false duplicate. The focused Rust edge regression passed.
- [x] Distinguish missing weights from negative weights in network
  validation. Null weights trigger the missingness check without a
  false bounds failure. The focused Rust edge test passed.
- [x] Flag zero-weight edges as incompatible with the positive-weight
  Kairos graph contract while retaining them in generic datasets.
  The Rust regression verifies that both compatibility warnings can
  appear together.
- [x] Qualify retention of zero-weight edges and Kairos compatibility
  warnings through Python Polars and CSV ingestion, Julia and installed R.
  Python and Julia also reproduced the checksummed artifact on replay.
  All 82 Rust boundary and 103 Python native tests passed, along with
  the complete Julia and freshly installed R native suites. The tested
  Python wrapper and extension matched the rebuilt CPython 3.15.2 wheel.

- [x] Reject adjustment options in the descriptive treatment contrast.
  Requests for matching or confounder adjustment previously returned
  the same unadjusted result. The Rust boundary regression failed before
  validation was added and passed after. Adjusted estimators remain open.

- [x] Add native empirical stratum standardization with arm counts and
  sample weights. The unequal-size analytic fixture returns 6, and
  strata without both arms fail explicitly. All 82 Rust boundary tests
  passed. Binding runtime qualification and broader estimators remain open.

- [x] Qualify empirical stratum standardization through CPython 3.15.2,
  Julia and installed R. Each binding returned the analytic contrast of 6
  with a low-stratum weight of 3/7 and rejected a stratum missing one arm.
  The wheel and C ABI library were rebuilt locally with stripping disabled
  after the default linker invocation exhausted disk space.

- [x] Add the treated-population estimand to native stratum standardization.
  The focused Rust regression verifies an ATT of 7.25, omission of
  control-only strata, and rejection of treated strata without controls.

- [x] Verify both stratified estimands through rebuilt CPython 3.15.2,
  Julia and installed R. The treated-population contrast was 7.25 with
  a low-stratum weight of 1/4. Each binding rejected a treated stratum
  without controls. Builds were staged sequentially to fit local disk space.

- [x] Report conditional standard errors for stratified contrasts with
  independent observations and fixed weights. The analytic Rust fixture
  verifies squared weighting across strata and a null result for a
  singleton arm. Binding runtime checks remain open for this field.

- [x] Verify stratified conditional standard errors through rebuilt
  Python 3.15.2, Julia and installed R. Both estimands returned the
  analytic sqrt(0.5) with replicated arms and null with singleton arms.

- [x] Expose native treatment contrast discovery with separate unadjusted
  and stratified capabilities. The focused Rust boundary test passed.

- [x] Verify treatment contrast discovery through rebuilt Python 3.15.2,
  Julia and installed R. Each returned the unadjusted method and the
  stratified method’s required inputs and ATE/ATT estimands.

- [x] Report treated fractions in included strata and count controls
  excluded from the ATT target population. The focused Rust causal
  boundary test passed.

- [x] Verify overlap diagnostics in Python 3.15.2, Julia and installed R.
  All three returned a treated fraction of 1/3 and counted one excluded
  control without changing the ATT. The focused causal tests passed.

- [x] Add normalized inverse-probability weighting for supplied propensities.
  The Rust analytic fixture returns -4 and rejects a zero propensity.
  Propensity estimation, uncertainty, discovery and binding checks remain open.

- [x] Expose supplied propensity weighting in treatment contrast discovery.
  Rebuilt Python 3.15.2, Julia and installed R each discovered the method,
  recovered the analytic estimate of -4 and rejected zero propensity.

- [x] Add per-arm weight concentration diagnostics for propensity
  weighting. The focused Rust fixture verified two observations,
  maximum normalized weight 0.8 and effective sample size 25/17.

- [x] Verify propensity weight concentration across rebuilt Python 3.15.2,
  Julia and installed R. Each returned 0.8 for the largest normalized
  weight and 25/17 for effective sample size in both arms.

- [x] Add the propensity-weighted ATT and advertise both estimands.
  The focused Rust fixture verified -41/17 and equal treated weights.
  Binding runtime qualification remains open for the weighted ATT.

- [x] Qualify supplied-propensity ATT through rebuilt Python 3.15.2,
  Julia and installed R. All three advertised both estimands and
  returned -41/17 with a maximum treated weight of 0.5.

- [x] Add a fixed-propensity sandwich standard error to normalized
  weighting. The focused Rust test verified sqrt(20.48) and null for
  singleton arms. Binding validation and fitted-propensity inference remain open.

- [x] Verify fixed-propensity sampling error through rebuilt Python
  3.15.2, Julia and installed R. Each returned sqrt(20.48) and null
  for singleton arms. Fitted-propensity inference remains open.

- [x] Estimate empirical propensities by caller-supplied stratum.
  The Rust regression checks input ordering, rejects absent overlap,
  and verifies that feeding the fitted probabilities into weighted ATE
  recovers the stratified estimate of 6. Binding qualification remains open.

- [x] Verify the empirical propensity fitting and weighted ATE workflow
  through rebuilt Python 3.15.2, Julia and installed R. Each preserved
  input ordering and recovered the analytic estimate of 6.

- [x] Add native scalar-score nearest-neighbour ATT with replacement.
  The focused Rust fixture verifies 2.5, deterministic ties and caliper
  failure. Binding qualification and matching inference remain open.

- [x] Qualify nearest-score matching through rebuilt Python 3.15.2,
  Julia and installed R. Each returned ATT 2.5, the same zero-based
  control indices and a clear caliper failure for an unmatched row.

- [x] Advertise nearest-score matching as ATT with a required caliper.
  Python 3.15.2, Julia and installed R discovered and invoked the
  advertised method, recovering 2.5 with deterministic matched pairs.

- [x] Report unique controls and effective control sample size for
  nearest-score matching. The Rust fixture checks both distinct and
  reused controls. Binding qualification and matching inference remain open.

- [x] Verify matched-control concentration through rebuilt Python 3.15.2,
  Julia and installed R. Two distinct controls gave effective sample
  size two, while one reused control gave effective sample size one.

- [x] Add opt-in caliper matching for the supported treated subset.
  The focused Rust fixture preserves strict default rejection, returns
  a subset estimate of 2 and reports excluded treated index 3.

- [x] Qualify partial caliper matching through rebuilt Python 3.15.2,
  Julia and installed R. Each returned matched_subset_att of 2 and
  excluded index 3, while the same request in strict mode failed.

- [x] Propagate supplied independent outcome variances through fixed
  matched pairs. The focused Rust fixture verifies sqrt(4.5) when two
  treated rows share a control. Binding and broader inference remain open.

- [x] Verify fixed-match outcome uncertainty through rebuilt Python
  3.15.2, Julia and installed R. The shared-control fixture returns
  sqrt(4.5), and omitting outcome variances returns null.

- [x] Fit ridge logistic treatment propensities for numeric covariates.
  The Rust fixture verifies analytic probabilities 0.25 and 0.75 and
  rejects a single treatment arm. Binding checks and fitted inference remain open.

- [x] Verify ridge logistic propensity fitting through rebuilt Python
  3.15.2, Julia and installed R. All three preserved the single-column
  matrix and recovered probabilities 0.25 and 0.75 with scale 2.

- [x] Score held-out covariate rows with saved ridge logistic fits.
  The focused Rust test recovers 0.5, 0.75 and 0.25 in requested order
  and rejects an invalid saved scale. Binding validation remains open.

- [x] Verify saved logistic propensity fit replay through rebuilt Python
  3.15.2, Julia and installed R. Each returned 0.5, 0.75 and 0.25 in
  the requested held-out row order.

- [x] Score held-out propensity predictions using Brier score and log
  loss. The Rust test verifies analytic values and rejects boundary
  probabilities. Binding runtime checks remain open.

- [x] Verify held-out propensity scoring through rebuilt Python 3.15.2,
  Julia and installed R. Each scored saved-fit predictions with analytic
  Brier score 0.125 and log loss -(log(0.5) + 2 log(0.75))/3.

- [x] Cross-fit ridge logistic propensities with caller-supplied folds.
  The Rust fixture verifies input order and rejection of a split that
  leaves a training complement with only one treatment arm.

- [x] Verify cross-fitted propensities through rebuilt Python 3.15.2,
  Julia and installed R. An interleaved fixture with different fold
  scales verifies complement-only scaling and original row order.

- [x] Add augmented inverse-probability weighted ATE with supplied
  nuisance predictions. The Rust fixture verifies the residual correction,
  an estimate of 4, standard error 2 and rejection of misaligned inputs.

- [x] Verify augmented ATE through rebuilt Python 3.15.2, Julia and
  installed R. Each recovered estimate 4 and influence standard error 2.

- [x] Add augmented ATT using supplied control predictions and
  propensities. The focused Rust fixture verifies estimate 2 and
  influence standard error 2. Binding qualification remains open.

- [x] Verify augmented ATT through rebuilt Python 3.15.2, Julia and
  installed R. Each recovered estimate 2 and influence standard error 2
  while retaining the augmented ATE assertions.

- [x] Compose cross-fitted stratum nuisance estimates with augmented ATE.
  The Rust fixture verifies complement-only outcome predictions, ATE 4
  and failure when a training complement lacks overlap. Binding checks remain open.

- [x] Verify cross-fitted stratum ATE through rebuilt Python 3.15.2,
  Julia and installed R. Each reproduced both sets of held-out
  predictions in input order and the augmented estimate of 4.

- [x] Validate cross-fitted adjustment on a synthetic two-stratum design
  with unequal treatment rates. Adjusted ATE recovers 6 while the raw
  contrast is 137/12. All 112 Python native and 82 Rust boundary tests passed.

- [x] Add cross-fitted stratum ATT. The Rust two-stratum fixture
  distinguishes ATT 7.25 from ATE 6 under unequal treatment prevalence.
  Binding runtime qualification remains open.

- [x] Verify cross-fitted stratum ATE and ATT through rebuilt Python
  3.15.2, Julia and installed R. The same unequal-prevalence fixture
  returns ATE 6 and ATT 7.25 in all three bindings.

- [x] Advertise augmented and cross-fitted stratum contrasts in native
  discovery, including the ATE-only treated prediction requirement.
  Rust metadata assertions and focused Python, Julia and R regressions passed.

- [x] Support cluster-robust influence standard errors for augmented
  contrasts. The focused Rust fixture verifies standard error 2 with
  two clusters and rejects a single cluster. Binding checks remain open.

- [x] Verify clustered augmented ATE through rebuilt Python 3.15.2,
  Julia and installed R. All three returned two clusters, standard
  error 2 and the same point estimate of 4 as independent-row inference.

- [x] Pass cluster labels through cross-fitted stratum ATE and ATT.
  The focused Rust test verifies the clustered ATE standard error of 2
  and rejects clusters split across folds. Binding checks remain open.

- [x] Verify clustered cross-fitted ATE through rebuilt Python 3.15.2,
  Julia and installed R. Each returned standard error 2 and rejected
  clusters assigned to more than one fold.

- [x] Verify clustered cross-fitted ATT through rebuilt Python 3.15.2,
  Julia and installed R. Each returned ATT 4 and standard error 2
  for two clusters held intact across the training splits.

- [x] Estimate balanced-panel event-study contrasts from unit outcomes.
  The focused Rust fixture verifies a zero pretrend, reference period,
  post-treatment contrast 4 and standard error 1. Binding checks remain open.

- [x] Verify balanced-panel event study through rebuilt Python 3.15.2,
  Julia and installed R. All three preserved unit-by-period ordering,
  flagged the reference period and returned estimate 4 and error 1.

- [x] Estimate staggered cohort-period ATT using not-yet-treated
  controls. The Rust fixture recovers 4 before the comparison cohort
  adopts and preserves null estimates after overlap ends. Binding and
  staggered inference qualification remain open.

- [x] Verify staggered cohort-period ATT through rebuilt Python 3.15.2,
  Julia and installed R. Each returned estimate 4 while the comparison
  cohort remained untreated and preserved nulls once no controls remained.

- [x] Add pointwise staggered ATT uncertainty for independent units.
  The focused Rust test verifies standard error 1 with replicated
  arms and null for singleton arms. Binding and joint inference remain open.

- [x] Verify staggered pointwise standard errors through rebuilt Python
  3.15.2, Julia and installed R. Each returned error 1 with replicated
  arms, null for singleton arms, and retained the never-treated control.

- [x] Summarize staggered ATT by event-time offset and cohort size.
  The Rust fixture verifies 8/3 for unequal cohort sizes, records
  unsupported cohorts and preserves null when none are supported.

- [x] Verify staggered event-time aggregation through rebuilt Python
  3.15.2, Julia and installed R. Each recovered 8/3 for unequal cohort
  sizes and preserved unsupported cohort counts and null estimates.

- [x] Report event-time standard errors that account for reused units.
  The Rust fixture verifies the shared-control covariance contribution
  and null for singleton arms. Binding and simultaneous inference remain open.

- [x] Verify shared-control event-time uncertainty through rebuilt
  Python 3.15.2, Julia and installed R. All three reproduce the analytic
  standard error and return null when a contributing arm is a singleton.

- [x] Return opt-in event-time covariance for staggered ATT. The Rust
  fixture verifies the pointwise variance on the diagonal and symmetric
  nonzero covariance across offsets. Binding and simultaneous bands remain open.

- [x] Verify event-time covariance through rebuilt Python 3.15.2, Julia
  and installed R. Each returned the analytic matrix in event-time order.
  The Python regression also checks null entries when uncertainty is absent.

- [x] Compute simultaneous staggered event-time bands using seeded
  unit multipliers. The focused Rust test verifies replay and the
  analytic max-t critical value. Binding runtime checks remain open.

- [x] Verify staggered simultaneous event-time bands through rebuilt
  Python 3.15.2, Julia and installed R. Each reproduced the seeded
  critical value sqrt(2) and the analytic interval endpoints.

- [x] Add a balanced event-time horizon for staggered ATT. The Rust
  fixture holds cohort composition fixed and recovers 4 at onset and
  2 at the next offset, with one cohort excluded. Binding checks remain open.

- [x] Verify balanced event-time horizons through rebuilt Python 3.15.2,
  Julia and installed R. Each returned onset ATT 4 and next-offset ATT 2,
  excluded one cohort from the summary and preserved all cohort-period rows.

- [x] Add staggered pre-adoption placebo contrasts with a common
  cohort baseline. The Rust fixture verifies an estimate of 1 and
  excludes a control already treated by the baseline. Bindings remain open.

- [x] Verify staggered placebo contrasts through rebuilt Python 3.15.2,
  Julia and installed R. Each returned estimate 1 and exactly one control
  still untreated at the baseline.

- [x] Support a never-treated comparator for staggered contrasts and
  their inference. The focused Rust fixture verifies onset ATT 3 with
  one eligible control. Binding runtime checks remain open.

- [x] Verify both staggered control selections through rebuilt Python
  3.15.2, Julia and installed R. Each returned onset ATT 3.5 with two
  not-yet-treated controls and ATT 3 with one never-treated control.

- [x] Accept labeled long-format panels for staggered ATT. The Rust
  fixture sorts shuffled rows, returns the expected estimate of 4 and
  rejects duplicate unit-period keys. Binding validation remains open.

- [x] Verify shuffled long-panel ATT through rebuilt Python 3.15.2,
  Julia and installed R. Each returned sorted unit labels and ATT 4.
  The Python fixture used Polars columns and checked duplicate rejection.

- [x] Expose a Polars DataFrame and LazyFrame entry point for staggered
  ATT. The focused Python test verifies equality with the native column
  request, including labels and cohort-period results.

- [x] Build a fresh macOS CPython 3.15 wheel with the Polars entry point.
  Verified Python 3.15.2 imported both wrapper and extension from the
  extracted wheel. All three focused panel tests passed, including eager
  and lazy Polars frames. Extraction required reclaiming the local Python
  build cache after the first attempt exhausted disk space.

- [x] Rank selected scenario grid paths by final adoption. The Rust
  boundary test verifies stable ties, places 1/1/3 and isolation of an
  invalid ranking from the executions. Binding checks remain open.

- [x] Verify scenario grid ranking through rebuilt Python 3.15.2, Julia
  and installed R. All three returned the same tied scenario order and
  places 1, 1 and 3.

- [x] Rank scenario grids by adoption area over a common horizon.
  The Rust regression verifies trapezoidal scoring alongside existing
  final-adoption ties and ranking error isolation. Binding checks remain open.

- [x] Verify adoption-area ranking through rebuilt Python 3.15.2, Julia
  and installed R. All three matched the analytic trapezoidal area on
  an uneven time grid and retained the final-adoption tie checks.

- [x] Rank scenario grids by interpolated time to a shared threshold.
  The Rust regression verifies tied crossings and null ranks for paths
  that never cross. Binding qualification remains open.

- [x] Verify threshold ranking through rebuilt Python 3.15.2, Julia
  and installed R. All three recovered the interpolated crossing,
  shared first place and null time and place for the unreached path.

- [x] Compare paired ensembles by integrated adoption difference.
  Four Rust scenario grid tests passed, including the analytic area
  and rejection of an area request with a single evaluation time.

- [x] Verify paired adoption-area comparisons through rebuilt Python
  3.15.2, Julia and installed R. All three reproduced opposite area
  differences, a zero mean and the shared evaluation horizon.

- [x] Report pointwise paired ensemble path uncertainty. Four Rust
  scenario-grid tests passed, including equality with the existing
  endpoint summary at the final sample. Binding checks remain open.

- [x] Verify paired path summaries through rebuilt Python 3.15.2, Julia
  and installed R. All three preserved sample order and matched the
  endpoint mean, standard error and empirical interval at the last sample.

- [x] Report paired path mean covariance in sample order. All four
  focused Rust scenario tests passed. The fixture verifies the endpoint
  variance and symmetric nonzero covariance. Binding checks remain open.

- [x] Verify paired path covariance through Python 3.15.2, Julia and R.
  The grid ranking fixtures compare the complete matrix with the outer
  product of pointwise standard errors for two opposite paired paths.
  The focused Python test, all 16 Julia assertions and the installed R
  package assertions passed against rebuilt native libraries. The Python
  wheel built successfully. The runtime check loaded the build artifact
  directly after wheel extraction ran out of disk space.

- [x] Verify the covariance size boundary through Python 3.15.2.
  The focused grid test accepts a 128 × 128 matrix, returns a typed
  comparison error at 129 sample times, and still returns all 129
  pointwise rows when covariance is disabled.

- [x] Document paired covariance axes, estimator, diagonal and
  over-limit behavior in the public API reference. The pointwise
  empirical intervals describe the observed replicate effects.

- [x] Center paired covariance on anchored differences to retain
  variation between adjacent floating-point effects. Covariance and
  pointwise standard errors now share the centering calculation.
  The focused Rust scenario unit tests passed, including agreement
  between variance and squared standard error for adjacent effects.

- [x] Requalify the paired covariance centering change (d8ad3c0f)
  through the Python 3.15.2 native extension, Julia and installed R.
  The focused Python grid test and all 16 Julia grid assertions passed.
  The R grid assertions exited successfully. Maturin built the wheel,
  and the Python fixture loaded the extension directly from the build.

- [x] Document the paired percentile and exact sign-flip options,
  assumptions and limits. The focused Python 3.15.2 grid fixture
  passed with four extreme assignments out of four for a zero
  observed mean and the expected seeded percentile endpoints.

- [x] Verify the paired percentile bootstrap and exact sign-flip
  response in Julia and installed R. With the same seed and two opposite
  effects, both reproduced the analytic percentile endpoints and all four
  extreme assignments. All 21 focused Julia assertions and the R grid
  assertions passed against the rebuilt Rust FFI library.

- [x] Verify finite-sample sign-flip calibration under independent,
  equiprobable signs. Across all 64 sign patterns for six distinct
  powers-of-two magnitudes, each attainable two-sided p-value occurred
  twice. The 5% rule rejected 2 of 64 assignments (3.125%), the exact
  rate for this discrete null. All four focused Rust scenario unit
  tests passed. This fixture covers one fixed magnitude design.

- [x] Preserve tied sign assignments when scaling effects for
  enumeration. Dividing [-10, -6, 10] by its maximum magnitude
  previously counted six extreme assignments instead of eight.
  Power-of-two scaling recovers all eight while bounding the sum.
  All five focused Rust scenario unit tests passed.

- [x] Rerun paired inference binding checks on 9c91630b. The
  focused Python 3.15 grid test, all 21 Julia assertions and the
  installed R grid assertions passed with rebuilt native libraries.
  The tied-effects regression itself is exercised by the Rust unit
  test. Maturin built the wheel and Python loaded the extension
  directly from the build.

- [x] Run the full Python native consumer test file against the
  extracted macOS arm64 wheel built from 9c91630b. The repo-local
  Python 3.15.2 imported the extension from the extracted wheel,
  and all 101 tests passed in 58.59 seconds. Removing generated
  Python build artifacts freed space for extraction.

- [x] Run the complete Julia and installed R native consumer files
  against the rebuilt Rust FFI library from 9c91630b. Both runners
  exited successfully. Together with the extracted Python wheel run,
  this covers the three local consumer suites on the same native source.

- [x] Run the complete locked Rust core suite on 03bb7474, whose
  source matches the three local consumer runs. All 273 unit and
  integration tests passed with zero failures or ignored tests.

- [x] Stabilize historical adopter categories on irregular time grids.
  The focused Rust regression preserved the categories under both a
  narrow translated grid and a grid whose raw endpoint span overflows.
  Classification still weights the supplied samples. Binding
  requalification remains open.

- [x] Verify the irregular adopter grids on 59b0e856 through the
  extracted Python wheel, Julia and R's raw JSON entry point. Both
  grids returned Early Adopters, Early Majority and Laggards.
  The Rust Polars historical inclusive-boundary regression also passed.
- [x] Preserve adjacent binary64 timestamps in R's structured request
  path. The ordinary R helper collapsed the narrow grid during JSON
  serialization. The raw JSON entry point accepted the same grid when
  formatted with 17 significant digits.
  The structured bridge now requests 17 significant digits for
  JSON serialization. A clean R package install passed the full native
  consumer file, including exact timestamp readback for both grids.

- [x] Retain automated Python and Julia consumer regressions for the
  narrow and overflowing adopter time grids, matching the installed R
  fixture. The focused Python test and all four Julia assertions
  passed, including exact timestamp readback and all category labels.

- [x] Preserve inclusive category boundaries on finite nonbinary
  spans. Power-of-two scaling recovered the exact thresholds for
  weights [0, 1, 0, 1] on both the original and translated grids.
  Both focused Rust core tests and the Polars historical boundary test
  passed, including the previously qualified overflowing-span fixture.

- [x] Check finite-span inclusive adopter boundaries on fe565142
  through rebuilt Python 3.15.2 and Julia and the installed R bridge.
  All three returned the four expected categories on both the original
  and translated grids. Python loaded the rebuilt extension directly.

- [x] Retain the inclusive finite-span boundary case in all three
  native consumer suites. The focused Python, Julia and installed R
  checks passed for the original and translated grids and retained
  the adjacent and overflowing timestamp checks.

- [x] Exercise the Julia IPC rewrite wrapper in the native consumer
  suite. The paired outcome file rewrote with the expected leading
  continuation marker, a second rewrite produced identical bytes for
  this fixture, and malformed input raised an error. All five focused
  Julia assertions passed. Direct Arrow.jl readback of an untouched
  Python Polars file remains open.

- [x] Retain a runnable Julia typed readback check for portable
  Polars IPC. With Arrow.jl 2.8.1, rewritten policy metadata and
  treatment contrast files matched their stream references across
  22 assertions covering column order, element types, values and
  missing entries. Run evidence/check-portable-arrow.jl with the
  original file and equivalent stream as arguments.

- [x] Generate the typed Arrow handoff fixtures from source with
  evidence/generate-portable-arrow.py. Python 3.15.2 and Polars 2.1.0
  produced a file and equivalent stream with nullable nested values,
  empty lists, Unicode and Int32 identifiers. The Julia readback check
  passed all eight assertions on the generated pair.

- [x] Reject contradictory migration ledger entries that claim
  qualification while recording remaining work. A synthetic complete
  ledger passed the release check and the same ledger with one open
  qualification item was rejected. The real development ledger passed
  validation and the release check still returned 1 for 38 obligations.

- [x] Retain the contradictory-ledger regression in the migration
  policy suite. All nine cases passed under repo-local Python 3.15.2.

- [x] Qualify continuous Kairos hazard forcing through a translated
  network scenario and its saved replay request. The Rust regression
  passed with a request at tick 3, adoption at tick 4 and an identical
  replay result. Replay starts from the original request and seed.

- [x] Share the Kairos hazard replay request as a Rust integration
  fixture and run it through Python 3.15.2, Julia and installed R.
  Each binding reproduced the [0, 0, 0, 1] adoption counts and an
  identical simulation result from its returned replay request.

- [x] Report RNG checkpoint scope consistently in Kairos network
  output. The focused network replay test passed with the scope field.
- [x] Implement stateful Kairos resume at dispatched adoption or
  service-completion boundaries. The Bass DES, homogeneous ABM and
  network checkpoint tests compare resumed and uninterrupted outcomes.
  The network checkpoint restores pending services, retries, deadlines,
  policy state and accounting. Python, Julia and R passed the scoped
  round-trip fixtures recorded below.
- [x] Qualify time-based pauses between dispatched events, including
  an outstanding hazard draw or integrated budget. Broader scenario
  and scientific qualification remains open.

- [x] Verify the emitted finite-population DES RNG checkpoint restores
  through the pinned Kairos API. After four adoptions the restored
  checkpoint and next four draws exactly matched a stream advanced
  through the corresponding four hazard distributions. The focused
  Rust integration test passed.

- [x] Isolate the finite-population Kairos adoption clock so its time,
  forcing cursor and RNG can be captured together at an event
  boundary. All 13 Rust Kairos integration tests passed, including
  forcing, replay and RNG checkpoint restoration.

- [x] Verify internal adoption-clock continuation after serializing
  the tick, forcing cursor and RNG at a dispatched event boundary.
  The restored clock matched the uninterrupted ticks and final RNG
  checkpoint after crossing a zero-hazard interval. The focused Rust
  unit test passed. Public simulation checkpoint restore remains open.

- [x] Expose event-boundary pause and resume for finite-population
  Bass DES. The Rust fixture paused twice, then matched the complete
  uninterrupted result and rejected a changed seed. The checkpoint
  binds effective forcing, population, tick scale, horizon and Kairos
  revision. Cross-binding qualification, ABM and network resume remain
  open.

- [x] Round-trip the Bass DES checkpoint through Python 3.15.2,
  Julia and installed R. R's JSON round-trip changed the numeric
  representation of the configuration object. A SHA-256 string
  now binds the effective Rust configuration across consumers.
  Each binding matched its uninterrupted result after one pause.
  Large tick values and multi-engine resume remain open.

- [x] Preserve large DES checkpoint ticks as decimal strings. The
  focused Rust resume test passed. Installed R round-tripped ticks
  49148533472158968 and 67101675129204106 through its structured
  bridge and matched the uninterrupted simulation result.

- [x] Extend adoption-boundary resume to homogeneous Bass ABM.
  The checkpoint retains susceptible ordering and the identity RNG.
  The restored trajectory matched every final agent state and the
  uninterrupted adoption history with a nonzero initial population
  of adopters. Duplicate susceptible indices were rejected. All 15
  Kairos Rust integration tests passed. Binding and network resume
  qualification remain open.

- [x] Round-trip homogeneous ABM checkpoints through Python 3.15.2,
  Julia and installed R. R's numeric JSON path could not preserve
  the identity RNG stream ID. The checkpoint now carries the identity
  RNG record as an opaque JSON string. Each binding reproduced the
  full uninterrupted result. The three Rust checkpoint tests passed.

- [x] Consolidate Kairos network resource accounting for a future
  event-boundary checkpoint. The state now owns the accumulated
  occupancy, capacity, wait totals and the ticks needed to extend
  those totals. All 29 Rust network integration tests passed.

- [x] Group network wait and retry state for checkpoint capture.
  Queue arrival times, active deadlines, retry wake ticks and retry
  counts now share one owner. Kairos still orders and admits waiters.
  All 29 Rust network integration tests passed.

- [x] Store network service admissions as typed records with agent,
  start tick and scheduled completion tick. Admission order can be
  used to reconstruct pending Kairos completions at a pause boundary.
  All 29 network tests passed with the existing result fields.

- [x] Verify reconstruction of tied Kairos completion order from
  network service admissions. Re-enqueueing admissions in their
  recorded order preserved dispatch order when two different service
  durations ended on the same tick. The focused unit test and all 29
  network integration tests passed.

- [x] Verify Kairos waiter reconstruction from queue arrival times
  and priorities. The rebuilt resource served the highest priority
  first and preserved arrival order for equal priority. Both network
  unit tests and all 29 network integration tests passed.

- [x] Reconstruct the Kairos reneging queue from active waiter
  arrival times and stored deadlines. Two waiters with different
  wait limits and a tied deadline expired in arrival order. All
  three network unit tests and all 29 network integration tests passed.

- [x] Implement network pause and resume at service completion
  boundaries. The focused Rust test paused with two services active
  and waiters queued, restored pending events and policy state, and
  matched the complete uninterrupted response. A changed seed was
  rejected. Broader checkpoint validation, cross-binding checks and
  scientific qualification remain open.

- [x] Qualify the shared network checkpoint fixture through the
  extracted Python 3.15.2 wheel, Julia and installed R. Each consumer
  paused after the first completion and restored its full result
  exactly, including the queue, service and policy traces. The Rust
  integration test also passed with the shared fixture. This verifies
  round-trips within each binding.

- [x] Exchange a Python-generated network checkpoint with Julia and
  installed R. R initially rejected the configuration because an exact
  integral float was serialized as an integer. Normalize exact integral
  JSON floats within the binary64 safe-integer range for fingerprinting.
  Both consumers then reproduced their uninterrupted result, and the
  focused Rust regression passed with the alternate numeric encoding.

- [x] Reject network checkpoints with backward retry or deadline
  ticks, incompatible RNG streams, invalid service durations or
  inconsistent resource totals. The regression exercises a stale
  retry and an impossible occupancy total. All 30 Rust network
  integration tests passed.

- [x] Verify chained network checkpoints at every completion in the
  scheduled resource fixture. Each checkpoint continued to the next
  completion and independently reproduced the full uninterrupted
  response when resumed to the horizon. Both focused Rust checkpoint
  tests passed.

- [x] Verify network checkpoint continuation with a continuous hazard
  ramp and nonzero imitation on a six-agent graph. The resumed run
  matched the uninterrupted event traces, resource accounting and RNG
  state. All three focused Rust network checkpoint tests passed.

- [x] Preserve the internal Bass hazard draw at a tick cutoff.
  The clock restores the pre-draw RNG state if the next decision
  crosses the cutoff. Continuing across a future forcing boundary
  matched the uninterrupted event tick and RNG checkpoint. Both
  clock unit tests and all 15 Kairos integration tests passed.
  Public tick-cutoff requests remain open.

- [x] Expose tick-based checkpoints for Bass DES and homogeneous ABM.
  The Rust fixture paused before the first adoption, paused again at
  an adoption after a forcing change, and reproduced the uninterrupted
  results for both engines. All 16 Kairos integration tests passed.
  Binding requalification and network tick cutoffs remain open.

- [x] Expose tick-based network checkpoints. The continuous hazard
  fixture resumed from every integer tick below its horizon, including
  before the first completion. Each resumed response matched the
  uninterrupted result. All 32 network integration tests passed
  before expanding the cutoff fixture, and all three checkpoint tests
  passed with the expanded fixture. Binding qualification remains open.

- [x] Retain tick-zero checkpoint regressions for DES, homogeneous
  ABM and the network engine in the Python, Julia and R consumer
  suites. The focused Python test, six Julia assertions and installed
  R checks reproduced the uninterrupted result for every engine.
  Python used the rebuilt native extension from the wheel build.

- [x] Preserve the seed on paused scenario artifacts and give a
  resume diagnostic when ranking or comparing a paused scalar path.
  The translated network scenario paused under continuous forcing
  and resumed through its saved request to the uninterrupted result.
  The focused Rust boundary test passed.

- [x] Verify the saved paused scenario request reproduces the
  identical checkpoint string before continuing. The focused Rust
  boundary regression passed for continuous network forcing.

- [x] Validate recorded service admissions before restoring network
  checkpoints. The completion tick must equal the admission tick plus
  the service duration, and the wait must match the original request.
  The malformed-completion regression and all three focused Rust
  network checkpoint tests passed.

- [x] Reject network checkpoints with inconsistent waiter, cooldown
  or active-service agent states. The regression changes a queued agent
  to susceptible and verifies rejection. All 32 network tests passed.

- [x] Move the binding checkpoint regression to tick six to exercise
  populated adoption history and network resource state. Python
  3.15.2, Julia and installed R reproduced the full uninterrupted
  result for each engine against rebuilt native artifacts. Rust
  retains the network cutoff sweep beginning at tick zero.

- [x] Retain tick zero, four and six in every binding regression.
  Python, Julia and installed R matched uninterrupted results for all
  three engines at each cutoff. Julia passed 18 focused assertions.

- [x] Distinguish paused executions in Arrow scenario exports.
  The focused Rust Polars test preserved the checkpoint and row order
  while marking only the completed execution successful.

- [x] Verify a mixed completed and paused Kairos scenario export in
  Python Polars. The extracted Python 3.15.2 wheel preserved both
  execution rows and the checkpoint exactly, with success flags
  true and false respectively. The focused consumer test passed.

- [x] Verify Julia and R Arrow readback for mixed completed and
  paused scenario rows. Both preserved row order, success flags and
  checkpoint content. Exact equality of all R row fields required
  passing the native JSON response directly because re-encoding the
  decoded R list changed its representation.

- [x] Expose raw response selection on R's structured kernel call.
  A clean package install passed the focused regression and exported
  a paused scenario directly from its native JSON as Arrow IPC.

- [x] Read back the Arrow file from the installed R structured
  raw-response helper. Arrow recovered both scenario IDs, the true
  and false completion flags, and exact decoded equality of every
  exported execution field, including the opaque paused checkpoint.

- [x] Run the local consumer suites at the checkpoint and Arrow
  milestone. The extracted Python 3.15.2 wheel passed the full
  native consumer and migration policy files (113 tests). Julia
  passed 428 assertions across 58 test sets, and the full installed
  R native suite exited successfully. The native Rust sources match
  9ac7533f, with the R bridge installed from 2e492674.

- [x] Run the locked Rust core and Polars suites at the same native
  source revision as the checkpoint consumer milestone. All 299
  tests passed with zero failures or ignored tests on c7538fac.

- [x] Pass locked all-target Clippy for the Rust core and Polars
  crates with warnings denied. Use fixed-size chunks in the exact
  sign-flip calibration test. The focused test and formatting passed.

- [x] Advertise checkpoint controls through native scenario discovery.
  Baseline and intervention identify adoption-count and tick pauses.
  Network identifies completion-count and tick pauses. The focused
  Rust discovery test verified the family-specific capabilities.

- [x] Read native checkpoint discovery through Python 3.15.2,
  Julia and installed R. The focused checks recovered the distinct
  adoption and service-completion pause controls. Sequential builds
  after cleaning generated artifacts resolved the initial disk limit.

- [x] Advertise the partial matching estimand and the assumptions for
  fixed-match uncertainty through causal discovery. The focused Rust
  test verified the required outcome variances and independence
  assumptions. Broader matching inference remains open.

- [x] Report the caliper, largest retained score distance and matched
  treated fraction. The focused Rust causal test verified full matching
  and partial matching with half the treated observations retained.

- [x] Verify retained-match diagnostics through rebuilt Python
  3.15.2, Julia and installed R. All three returned a caliper of
  0.5, maximum retained distance zero and matched treated fraction
  0.5 on the same partial ATT fixture.

- [x] Report optional named covariate means and differences over
  retained matched pairs. Reused controls contribute once per pair.
  The Rust fixture matched the analytic 10 for controls with values
  0, 0 and 30, and rejected misaligned covariate arrays. Binding
  checks and broader matching inference remain open.

- [x] Verify matched covariate balance through rebuilt Python 3.15.2,
  Julia and installed R. Each returned treated mean 30, control mean
  10 and difference 20 for the reused-control fixture. Document the
  aligned covariate input and pair weighting.

- [x] Report the original group means alongside matched balance.
  The Rust reused-control fixture showed the covariate mean difference
  increasing from 15 before matching to 20 after matching.

- [x] Verify pre-match and matched covariate means through rebuilt
  Python 3.15.2, Julia and installed R. All three recovered the
  increase in the analytic fixture's mean difference from 15 to 20.

- [x] Report signed standardized covariate balance before and after
  matching with a shared pooled pre-match denominator. The analytic
  Rust fixture matched 15/sqrt(150) and 20/sqrt(150), and returned
  null for a constant covariate. Binding qualification remains open.

- [x] Verify standardized balance in rebuilt Python 3.15.2, Julia
  and installed R. All three recovered the two analytic standardized
  values and a null standardized difference for the constant covariate.

- [x] Retain the standardized balance and constant-covariate
  cases in Python's native consumer suite. Both focused matching
  tests passed on Python 3.15.2.

- [x] Preserve pooled pre-match spread for adjacent floating-point
  values by subtracting an anchor before scaling. The Rust unit
  regression matched the analytic spread within two machine epsilons,
  and the focused causal boundary test passed.

- [x] Retain an adjacent-value case in the Python matched-balance
  regression. The rebuilt native extension returned a finite
  standardized difference for the narrow covariate and null for
  the constant covariate. The focused Python 3.15.2 test passed.

- [x] Preserve narrow standardized mean differences by centering
  both groups on the same anchor before taking their means. The
  analytic adjacent-value numerator is one-sixth of a floating-point
  step. Both causal unit tests and the focused boundary test passed.

- [x] Verify the narrow standardized balance numerator through the
  rebuilt Python 3.15.2 extension. The focused consumer test checks
  the analytic nonzero pre-match value and zero matched value.

- [x] Retain the analytic narrow standardized balance case in Julia
  and R. Against rebuilt FFI, both matched the nonzero pre-match
  value, zero post-match value and null constant-covariate result.

- [x] Derive raw and standardized matched balance differences from
  the same centered numerator. Power-of-two scaling preserves exact
  ordinary fixture values while the narrow unit case retains the
  one-sixth-step difference. Both causal unit tests and the focused
  causal boundary test passed.

- [x] Verify the raw and standardized narrow-balance differences
  in Python 3.15.2, Julia and installed R against rebuilt native
  artifacts. Each binding retained the raw one-sixth-step difference.

- [x] Add fixed-prediction adjustment to nearest-score matching.
  The Rust fixture preserved the unadjusted ATT of 2.5 and recovered
  an adjusted ATT of 2. It also rejected misaligned predictions.
  Binding checks and fitted-model inference remain open.

- [x] Verify fixed-prediction adjusted matching through rebuilt
  Python 3.15.2, Julia and installed R. Each preserved the unadjusted
  estimate of 2.5 and returned the analytic adjusted estimate of 2.

- [x] Compose native cross-fitted stratum control predictions with
  adjusted matching. The focused Rust boundary fixture reused the
  returned row-aligned predictions and recovered an adjusted effect
  of 6 from an unadjusted effect of 5. Fitted-model uncertainty remains
  open.

- [x] Retain a Python consumer regression for the cross-fitted
  control prediction handoff. The native workflow reproduced the
  analytic unadjusted and adjusted effects without supplied variances.

- [x] Retain the fitted-control matching workflow in Julia and R.
  The installed consumers recovered the unadjusted effect of 5 and
  adjusted effect of 6, with null conditional uncertainty when
  outcome variances were absent.

- [x] Expose sparse outcome coefficients for matched ATT. The
  focused Rust fixture verified two treated weights of 0.5 and a
  reused control weight of -1. Binding checks remain open.

- [x] Use the reported sparse outcome weights to compute conditional
  matching uncertainty. The analytic Rust fixture still recovered
  sqrt(4.5) under independent outcomes with one reused control.

- [x] Reconstruct the matched estimate and independent-outcome
  conditional standard error from the exported weights in rebuilt
  Python 3.15.2, Julia and installed R. Each matched the native result
  for a reused control.

- [x] Propagate caller-supplied shared outcome variation through
  fixed matching. The Rust fixture canceled a common factor and
  added four units of variance for a treatment-only factor, yielding
  sqrt(8.5) with reused controls. Misaligned factors were rejected.
  Binding checks and fitted-model uncertainty remain open.

- [x] Verify shared outcome variation through rebuilt Python 3.15.2,
  Julia and installed R. All three returned sqrt(8.5) for the same
  reused-control fixture with common and treatment-only factors.

- [x] Identify the matched conditional uncertainty model and factor
  count in the result. The focused Rust fixture distinguishes the
  independent and two-factor cases while retaining both analytic errors.

- [x] Retain the correlated standard error and uncertainty model in
  Python's matching consumer test. The rebuilt Python 3.15.2
  extension returned sqrt(8.5) and reported two covariance factors.

- [x] Retain the correlated matching fixture in Julia and R.
  Both installed consumers recovered sqrt(8.5) and identified
  the two-factor covariance model with rebuilt FFI.

- [x] Compute the covariance factor contrast before rescaling to
  outcome units. The Rust boundary fixture canceled opposing
  maximum-magnitude loadings and returned zero conditional error.

- [x] Verify that extreme covariance loadings cancel when the
  contrast is finite and return a typed invalid request when the
  actual standard error overflows. The focused Rust test passed.

- [x] Propagate fitted stratum control-mean uncertainty through
  adjusted matching with fixed scores, folds and pairs. The Rust
  boundary fixture recovered adjusted effect 6 and standard error
  sqrt(3) from its analytic outcome coefficients, and rejected an
  unsupported training stratum. Binding validation and uncertainty
  from estimated scores remain open.

- [x] Verify fitted-stratum matching through rebuilt Python 3.15.2,
  Julia and installed R. All three returned adjusted effect 6 and
  adjusted standard error sqrt(3) with the same fixed matching design.

- [x] Center the fitted outcome contrast before accumulation so a
  shared shift cancels despite fractional-weight rounding. The Rust
  unit fixture canceled a maximum-magnitude common loading. All three
  causal unit tests and the focused causal boundary test passed.

- [x] Verify fitted-control outcome uncertainty in the Python consumer
  on Python 3.15.2. The fixed-design fixture checked analytic outcome
  weights, cancellation of a common covariance loading, and propagation
  of a treated-only loading into both effect standard errors.

- [x] Verify the same fixed-design fitted-matching covariance fixture
  through Julia and the installed R binding against a rebuilt Rust FFI
  library. Both bindings reproduced the Python estimates, outcome weights
  and standard errors.

- [x] Run cross-fitted ridge-logistic propensity estimation directly in
  native matching. Supply covariate rows, folds and a positive ridge via
  `fit_scores` in place of `scores`. The result includes the held-out
  probabilities and fold provenance. Four Rust causal unit tests passed.
  The standard errors still condition on the selected pairs. Score-fit
  uncertainty and binding qualification for this input remain open.

- [x] Verify the `fit_scores` matching input through rebuilt Python
  3.15.2 and the Julia and installed R bindings against a rebuilt Rust
  FFI library. Each reproduced a matched estimate of 5 and four held-out
  propensity scores of 0.5. These fixtures exercise the binding contract.

- [x] Report the scope of uncertainty for fitted-score matching and
  the minimum distance from an accepted match to the caliper boundary.
  The Rust fixture verified zero slack at the cutoff and positive slack
  after widening it. Propagation of score-estimation uncertainty remains
  open.

- [x] Verify fitted-score uncertainty scope and zero caliper slack in
  the Python, Julia and installed R consumers against rebuilt native
  libraries. The fitted-control covariance fixture also passed in each.

- [x] Report the minimum gap between the chosen control and its nearest
  alternative across accepted pairs, alongside the caliper slack. Rust
  tests verified positive separation and zero separation for duplicate
  scores and equidistant controls. This is a score sensitivity diagnostic.

- [x] Verify that Python, Julia and installed R expose a zero runner-up
  gap when two controls have the same fitted score. All three passed
  against rebuilt libraries. Local disk capacity required reusing the
  FFI build space for the Python link after the Julia and R checks.

- [x] Propagate caller-supplied joint score draws through native
  rematching. Preserve each draw's estimate, pairs and treated coverage
  so changes in the target subset remain visible. The Rust fixture
  recovered effects 4 and 2 after switching the selected control.
  Native score draw generation and binding qualification remain open.

- [x] Preserve conditional outcome errors per score draw and report an
  equal-weight mixture standard deviation when all draws target the
  same treated subset and supply conditional errors. The analytic
  fixture combined between-draw variance 1 and mean conditional
  variance 4 to recover sqrt(5). Six focused Rust causal tests passed.

- [x] Verify that score draws matching different treated observations
  leave the mixture uncertainty unset even when both draws retain the
  same number of observations. The Rust fixture preserved effects 4
  and 6 and their distinct unmatched indices. Seven causal tests passed.

- [x] Verify caller-supplied score-draw rematching and mixture uncertainty
  through Python, Julia and installed R against rebuilt Rust libraries.
  Each reproduced draw effects 4 and 2 and mixture spread sqrt(5).

- [x] Generate deterministic leave-one-out score refits in Rust from
  explicit training omission indices. Each omission is shared across
  fold fits and leaves evaluation rows in place. Rematching reports
  per-refit pairs and estimates. These refits measure sensitivity to
  training observations, so their mixture errors are unset. Eight
  focused Rust causal tests passed. Binding qualification remains open.

- [x] Verify leave-one-out score refits in Python, Julia and installed R
  against rebuilt native libraries. Each preserved the omitted training
  indices and the rematched control indices while leaving the mixture
  standard deviation unset even with supplied outcome variances.

- [x] Retain rematches that fail the caliper coverage requirement as
  individual insufficient-match results. Any failed draw suppresses
  mixture uncertainty, and malformed inputs still reject the request.
  Nine focused Rust causal tests passed.

- [x] Verify that Python, Julia and installed R preserve an insufficient
  score draw alongside a successful draw. All three returned estimate
  4 for the successful draw and an unset mixture standard deviation.

- [x] Generate seeded score bootstrap draws in Rust by resampling
  training observations within fold and treatment groups. Each draw
  shares the resample across fits and retains held-out evaluation rows
  for rematching. Ten focused Rust tests passed, including deterministic
  replay, changing matches and mixture output. Binding verification and
  empirical coverage calibration remain open.

- [x] Verify seeded score bootstrap through rebuilt Python, Julia and
  installed R consumers. Each replayed all eight draws with seed 17 and
  returned positive mixture spread. Empirical coverage remains open.

- [x] Measure synthetic coverage of a normal interval centered on the
  mean rematched effect with half-width 1.96 times the mixture spread.
  With fixed covariates [0,3,2,5,1,4,3,6], alternating treatment,
  contiguous four-row folds, ridge 1 and caliper 1, generate outcomes
  as 2*treatment + slope*covariate + independent unit Gaussian noise.
  Use Python random.Random(721) continuously across both slope arms,
  500 replications per arm, unit outcome variances, and 32 native
  bootstrap draws seeded by zero-based replication index. Coverage
  was 486/500 with slope 0 (mean error 0.0263) and 58/500 with slope 2
  (mean error 3.3803). The second arm deliberately has poor overlap.
  The mixture spread does not account for bias from poor overlap.
  General calibration and scientific acceptance remain open.

- [x] Repeat the poor-overlap experiment using the true fixed control
  predictions (twice each covariate value). Mean error fell from 3.3803
  to -0.0334 over the same 500 replications. Fixed predictions are
  deterministic offsets, so propagate each rematch's existing outcome
  variance into its adjusted standard error and the adjusted mixture.
  The analytic fixture verified adjusted spread 2. Ten causal tests passed.

- [x] Verify fixed-prediction uncertainty through rebuilt Python,
  Julia and installed R. Each returned adjusted mixture spread 2
  and unadjusted spread sqrt(5). Focused refit, insufficient-draw
  and seeded bootstrap checks also passed in all three consumers.

- [x] Measure coverage with independently estimated control predictions.
  Repeat the 500 slope-2 replications above using the same evaluation
  outcomes and score seeds. For each replication fit an intercept and
  slope to four independent control outcomes at covariates [0,1,2,3],
  generated with slope 2 and unit Gaussian noise from random.Random(122).
  Use the fitted slope times evaluation covariate as the prediction
  because the common intercept cancels. With intervals centered on the
  mean adjusted effect and half-width 1.96 times adjusted mixture spread,
  true predictions covered 477/500 (mean error -0.0334). Estimated
  predictions treated as fixed covered 451/500 (mean error -0.0202).
  Propagation of prediction-fit covariance remains open.

- [x] Propagate caller-supplied covariance factors for externally fitted
  control predictions whose errors are independent of outcome errors.
  Each rematch applies its contrast weights to the prediction factors.
  The Rust fixture canceled a common loading and recovered adjusted
  mixture spread sqrt(6), with raw spread still sqrt(5). Eleven causal
  tests passed. Binding checks and coverage with these factors remain open.

- [x] Verify independent prediction covariance through rebuilt Python,
  Julia and installed R using the analytic sqrt(6) adjusted and sqrt(5)
  raw fixture. Repeat the paired estimated-control simulation with
  the known slope-error loading x/sqrt(5). Coverage was 473/500
  compared with 451/500 when the predictions were treated as fixed.
  This evidence is limited to the specified independent Gaussian fit.

- [x] Extend SARIMA optimizer evidence to mixed ARMA(1,1) fits near
  positive and negative stationarity boundaries. With simulated AR
  coefficients +/-0.985 and MA coefficient 0.6, the Rust likelihood
  through installed R agreed with stats::arima within 1e-6 in both cases.
  The reproducible checker is check-mixed-near-unit-root.R. The release
  verifier still reports 38 open obligations.

- [x] Reproduce a seasonal optimizer gap with a mixed
  SARIMA(1,0,1)(1,0,1)[4] reference. With seasonal AR +0.97, Rust
  exceeded the R likelihood by 1.77e-6. With seasonal AR -0.97,
  Rust was lower by 0.02036 and maximum coefficient deviation was
  0.04254. Both R fits converged. The comparison script is
  compare-seasonal-boundaries.R. Optimizer qualification remains open.

- [x] Add paired coefficient moves when a SARIMA coordinate sweep stalls.
  Search all pairs for fits of up to eight coefficients and adjacent MA
  pairs above that limit. On the negative seasonal reference, the log
  likelihood shortfall fell from 0.02036 to 0.0000991 and the maximum
  coefficient error fell from 0.04254 to 0.00296. Existing mixed seasonal
  and mixed near-unit-root R checks passed. The residual gap remains open.

- [x] Compare unequal paired moves for SARIMA fits of up to eight
  coefficients. The negative seasonal likelihood shortfall fell from
  9.91e-5 to 2.56e-6 and maximum coefficient error fell from 0.00296
  to 0.000489. The positive case reproduced its prior result. The
  comparison exceeded two minutes in the local debug build. Reducing
  stationary covariance work and checking broader optimizer regressions
  remain open.

- [x] Double the stationary covariance summation horizon per iteration
  and restrict the expanded paired search to small models with both
  seasonal AR and MA terms. All six focused Rust SARIMA tests passed
  at their existing tolerances. The rebuilt R boundary comparison
  took 11.54 seconds in the debug build and the negative seasonal
  likelihood difference fell to 1.28e-8. The mixed and integrated R
  references also passed. Broader qualification remains open.

- [x] Enforce likelihood agreement within 3e-6 and coefficient
  agreement within 5e-4 for both difficult seasonal reference fits.
  The R regression check passed and the negative reference also
  passed through Julia and rebuilt Python using the same simulated
  values and R estimates. Broader optimizer qualification remains open.

- [x] Qualify time-to-threshold ranking when no scenario reaches the
  threshold. The Rust boundary check retained input order and null
  scores and places for all three scenarios. All 13 focused scenario
  tests passed. Direct untouched Polars file readback in Arrow.jl
  remains open alongside the existing native rewrite path.

- [x] Preserve finite adoption areas when midpoint terms would
  individually underflow or the raw time span would overflow. This
  calculation is shared by scenario ranking and ensemble comparison.
  The Rust fixture also verified rejection of an unrepresentable area.
  Six scenario unit tests and 13 scenario boundary tests passed.

- [x] Preserve subnormal adoption-area contributions across long time
  intervals by scaling the interval before halving the adoption sum.
  Six focused Rust scenario tests passed. The migration ledger retains
  the scenario registry at bounded native status, with cross-version
  replay and release parity still open. The verifier reports 38 open
  release obligations.

- [x] Reject unsupported top-level fields in embedded current-schema
  scenario replay requests. The scenario boundary fixture rejects an
  injected request-level model key, preserves same-version replay, and
  still classifies different kernel versions as unavailable. All 13
  scenario boundary tests passed. Cross-version replay remains open.

- [x] Run the Rust core integration milestone across all test targets.
  The initial run found stale exact-object matching assertions. Update
  both to assert the new caliper slack and runner-up gap as well as the
  original fields. All 301 unit and integration tests passed across
  the initial and resumed runs. Development policy validation passed
  with 38 release obligations still open.

- [x] Rebuild the native Python extension from faa2fc4a and run the
  entire native Python consumer test file on repo-local Python 3.15.2.
  All 112 tests passed in 60.65 seconds. The debug extension was loaded
  through the worktree test environment. Clean distribution evidence
  and the corresponding full Julia and R runs remain separate gates.

- [x] Run the complete Julia and R native consumer files against the
  debug Rust FFI library rebuilt from 1ace9cfb. Both exited zero.
  The first R invocation used a stale installed package and stopped
  at the paired-outcomes IPC helper. A fresh R CMD INSTALL into an
  isolated library resolved the mismatch and the full rerun passed.
  This qualifies the worktree bindings with an explicit FFI path.

- [x] Verify Julia bundled FFI discovery on 22ac2f49 in a staged package
  with INNOVATE_NATIVE_LIBRARY unset. The initial standalone stage
  lacked repository-relative test fixtures. Mirroring the source
  layout with copied Rust test fixtures resolved that test dependency.
  The complete Julia 1.13.3 suite passed with the debug library staged
  in deps. This is a local bundle check using the existing Julia depot.

- [x] Build and install the local macOS ARM64 R binary archive from
  30a51ae5 using R 4.3.0 and the current debug Rust FFI library.
  Archive SHA-256: 2f50fad649d4da7d0ffed16c02482dd28806ed56bbd6f3e3ab8f62b7d4240609.
  After reclaiming generated staging copies to resolve an out-of-space
  extraction, R CMD INSTALL into a fresh library succeeded. The full
  R consumer suite passed with INNOVATE_NATIVE_LIBRARY unset and the
  package path verified. Other platforms and release distribution
  qualification remain open.

- [x] Build a locked local macOS ARM64 CPython 3.15 wheel from 46f48aeb
  and install it into a fresh Python 3.15.2 environment.
  Wheel SHA-256: c3cd2619c82ea9027e6536ba2ffe0c6002f9fb154308ba674c90a718ab5a909d.
  With PYTHONPATH unset, the extension loaded from the installed wheel
  and all 112 native consumer tests passed in 59.40 seconds. Disk space
  required reclaiming a generated Julia library copy before linking.
  This is an unstripped debug wheel and local-platform evidence.

- [x] Bound in-process remote observability identifiers to 256 UTF-8
  bytes, reject control characters, and reject unknown envelope fields.
  The native boundary test passed for valid Unicode at the byte limit,
  oversized and multiline trace IDs, and an unsupported transport
  field. The first debug strip ran out of space, so the test binary
  was rebuilt without stripping. Hosted transport and identity remain open.

- [x] Enforce the in-process remote request limit while encoding the
  embedded request. The Rust test accepts exactly one million encoded
  bytes and rejects excess bytes introduced by JSON escaping. The remote
  dispatch boundary test also passed. Hosted transport must separately
  enforce limits before decoding requests.

- [x] Filter native registry discovery by a recognized extension point.
  Unfiltered listing still returns all registered manifests. The Rust
  boundary checks covered inclusion, exclusion, invalid point rejection,
  and duplicate registration. Extension execution remains open.

- [x] Add version-checked removal to the in-process native extension
  registry. Rust boundary tests verified removal, retry after removal,
  re-registration with a new version and rejection of a stale version.
  The check is atomic under the registry lock and compares versions.
  Hosted lifecycle and extension execution remain open.

- [x] Verify the in-process extension lifecycle through rebuilt Python,
  Julia and installed R. All three rejected stale version removal,
  listed the surviving version by extension point, removed it with
  the current version, and returned false on retry. The first FFI link
  ran out of space and succeeded after reclaiming generated test binaries.

- [x] Resolve a registered native extension by name, expected version
  and declared extension point. The lookup returns a manifest snapshot
  for a consumer and rejects stale, missing and undeclared selections.
  Both Rust extension boundary tests passed after reclaiming generated
  Python debug libraries for linking. Callable execution remains open.

- [x] Invoke a trusted Rust host callback using the validated extension
  selection and manifest snapshot. The callback runs after the registry
  lock is released, as verified by resolving the manifest again inside
  the callback. A stale selection never invokes it. Both Rust extension
  boundary tests passed. Binding callback APIs and per-point payload
  contracts remain open.

- [x] Expose trusted caller-supplied extension callbacks in Python,
  Julia and R. The bindings resolve the selection through Rust and
  pass its snapshot and the original payload to the callback. All
  three runtime checks invoked a valid callback and rejected a stale
  version before invocation. Python also rejected an extra model key.
  Per-point payload contracts and hosted qualification remain open.

- [x] Route dataset-provider callback results through native ingestion
  and return a validated dataset artifact with provenance and checksum.
  Rust, Python, Julia and R share the existing ingestion path. The
  Rust boundary tests verified valid artifact creation and rejection
  of missing provenance. All three Rust extension tests passed.
  Binding runtime qualification and other per-point contracts remain open.

- [x] Verify dataset-provider callbacks with the same adoption fixture
  through rebuilt Python, Julia and freshly installed R. Each returned
  a valid dataset artifact with checksum
  fada5d71f8981fa746be73b26de45ad470fa4e74f6a7d7a48234f4f50de0a76a
  and rejected callback output without dataset provenance.

- [x] Retain dataset-provider contract regression checks in all three
  native consumer suites. The focused Python, Julia and R checks passed
  with the same checksum and invalid-output rejection. Each test removes
  its registration after execution.

- [x] Verify a singleton network edge returned by an R dataset provider.
  The binding preserved source and target IDs "001" and "1" as distinct
  array values, kept the weight as a one-element array, and retained
  the provenance source as a scalar. The focused installed R check passed.

- [x] Define diagnostics callback output as an ordered array of named
  pass or fail checks with messages. Rust derives the aggregate status
  and failed names and rejects duplicate check names. The callback APIs
  use the same Rust contract. Three focused Rust extension tests passed.
  Binding runtime qualification for diagnostics remains open.

- [x] Verify diagnostics callbacks through rebuilt Python, Julia and
  freshly installed R. Each preserved the ordered checks, returned
  failed name b and ok=false, and rejected duplicate names. The R
  assertion compares JSON object fields independently of field order.
  The first FFI link ran out of space and passed after reclaiming a
  generated test binary.

- [x] Resolve model registry callback aliases against the Rust catalog.
  The output preserves alias order and includes each target capability.
  Rust rejects unknown targets and aliases that shadow native keys.
  The focused Rust unit test passed. Python, Julia and R route callback
  output through the shared operation. Binding checks and custom model
  dispatch remain open.

- [x] Verify model registry callbacks through rebuilt Python, Julia and
  freshly installed R. Each resolved baseline to the native Bass
  capability and rejected an alias attempting to shadow the Bass key.
  This covers alias discovery. Custom model dispatch remains open.

- [x] Revalidate dataset artifacts returned by serialization adapters
  through the native artifact ingestion path. The Rust boundary fixture
  round-tripped an unchanged artifact and rejected a changed adoption
  value with the original checksum. All three extension boundary tests
  passed. Binding qualification and other artifact formats remain open.
