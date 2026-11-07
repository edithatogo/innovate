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
- [ ] Improve automatic multi-product and lock-in starts or optimization.
  Both independent forced fixtures returned non-converged automatic fits
  at their existing 200-iteration budget. Alternative base seeds were
  investigated and discarded. The explicit-start conformance checks remain
  in place, and automatic-start qualification remains open for these models.

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
- [ ] Qualify parallel workloads in the installed bindings and measure
  throughput across representative model families.
