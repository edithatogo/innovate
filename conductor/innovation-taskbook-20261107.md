# Innovation Native Capability Implementation Plan

> For agentic workers: execute one ready task at a time using superpowers:executing-plans or superpowers:subagent-driven-development. Every task remains open. Read the owning track spec and this task's dependencies before editing.

**Goal:** Deliver a general innovation-modelling library with a mandatory Rust core, Polars/Arrow interfaces, validated scientific methods and a hardened scoped release.

**Architecture:** Native core plus data/interchange and optional Kairos/game adapters; Voiage consumes joint outputs through an optional integration boundary. Applications own their scenarios.

**Tech Stack:** Qualified Rust numerical dependencies, Polars/Arrow, native bindings, Kairos, isolated independent test oracles and versioned benchmark data.

**Spec:** `product.md`, `frontier-programme-20261107.md`, `execution-contract-20261107.md`, the owning track's `spec.md`, and linked research protocols.

## Global constraints

Rust owns scientific execution; no production Python fallback. Pandas/xarray are unsupported in the new release, including transitive supported extras. Root pins/shared types have one integrator. Current hybrid code is not retrospectively certified. Proposed paths and commands below become runnable when their prerequisite task creates the corresponding target. External oracle environments are separate from production installs. No task closes from a document alone.

## Review focus

Wrong stock/flow/transaction meanings (R02/M03/M04); invalid or stale posterior state (I01/I03); secretly falling back to Python (R05/R06/E01); composite non-identifiability and false-cycle detection (H01–H06); cyclic dependencies or broken joint uncertainty (E02/E03/G01). Each has named tests below.

## How to execute each packet

For every task: (1) create the named test with the stated assertions; (2) run the listed command and record the expected red result; (3) implement the stated interface in the listed files; (4) rerun that command plus affected contracts and record green output; (5) review, commit and attach a receipt. All five are mandatory even where the packet summarises them in one checklist. Commands using new test targets require creating those targets first; compilation failure alone does not substitute for a behavioural regression after an interface exists. Cross-repository tasks run in the named provider, never as copied implementation inside Innovate.

---

## Scientific integrity — scientific_integrity_20261107

### I01 — Invalid inference cannot look like a posterior
Dependencies: none. Modify `src/innovate/fitters/bayesian_fitter.py`, `src/innovate/fitters/blackjax_fitter.py`, `src/innovate/probabilistic.py`; create `tests/unit/test_inference_failure_semantics.py`.
Interface: existing fit/accessor methods retain explicit status until native replacement; failed refit clears previous draws.
Test: forced warmup failure, forced sampler failure and successful-fit-then-failed-fit yield invalid status; posterior/credible-interval access raises; no synthetic normal draws appear. An explicitly requested point estimate has no posterior label.
Command: `uv run pytest tests/unit/test_inference_failure_semantics.py -q`.
- [ ] Write/run red tests; remove fabricated-posterior fallback; propagate invalid status; rerun green; review and commit. This is a safety repair of legacy code, not continued production support after R05/R06.

### I02 — Honest estimator dispatch
Dependencies: none. Modify `src/innovate/causal/policy.py`, `src/innovate/causal/evaluation.py`, `src/innovate/reports/claims.py`; create `tests/unit/test_causal_estimator_dispatch.py`.
Interface: descriptive differences remain descriptive; unsupported matching/weighting/forest choices return an unsupported-method error rather than the same naive calculation.
Test: method selection changes a verified implementation or raises; unused requested confounders raise; empty groups/non-finite data fail; an unadjusted fixture returns the known difference only under its descriptive method.
Command: `uv run pytest tests/unit/test_causal_estimator_dispatch.py -q`.
- [ ] Run red/green cycle; document estimand and identification boundary; review and commit without inventing a new general causal framework.

### I03 — Native inference states and independent gradients
Dependencies: R01, R02, I01. Create `crates/innovate-core/src/inference/status.rs`, `crates/innovate-core/tests/inference_status.rs`, `crates/innovate-core/tests/gradient_oracle.rs`.
Interface: `FitOutcome::{Valid(FitResult),Invalid(FitFailure)}` and `posterior(&FitOutcome)->Result<&PosteriorDraws,InnovationError>`; `log_density(theta:&[f64], data:&ObservationSeries)->Result<(f64,Vec<f64>),InnovationError>`.
Test: invalid states cannot expose draws; analytic Gaussian log-density/gradient agree with independent formula and finite differences; zero/invalid variance is rejected. Reset does not retain prior results.
Commands: `cargo test --locked -p innovate-core --test inference_status --test gradient_oracle`.
- [ ] Create types/tests, record red, implement minimal native contract, record green, review and commit.

### I04 — Coverage and calibration harness
Dependencies: I03, A01. Create `validation/inference/protocol.json`, `validation/inference/run.py`, `tests/validation/test_inference_protocol.py`.
Interface: runner consumes immutable native executable plus simulator/seed manifest, emits per-replicate ranks, intervals, failure status and summary receipts; Python orchestrates only.
Test: injected biased estimates and fake narrow intervals are detected; failed replicates count in denominator; exact conjugate-normal fixture calibrates within prespecified Monte Carlo bounds.
Command: `uv run pytest tests/validation/test_inference_protocol.py -q`; execute `uv run python validation/inference/run.py --protocol validation/inference/protocol.json` only after native candidates exist.
- [ ] Pre-register replicate count/thresholds using Monte Carlo precision, run red/green harness tests, retain full results, review and commit; do not call non-rejection proof of universal calibration.

## Rust core — rust_core_convergence_20261107

### R01 — Extract a binding-independent native workspace
Dependencies: none. Modify `bindings/rust/src/lib.rs`, `bindings/rust/Cargo.toml`; create root `Cargo.toml`, `crates/innovate-core/Cargo.toml`, `crates/innovate-core/src/lib.rs`, `crates/innovate-core/tests/bass_reference.rs` and Cargo.lock.
Interface: `bass_cumulative(t:f64,p:f64,q:f64,m:f64)->Result<f64,InnovationError>`; bindings re-export rather than duplicate it.
Test: t=0 gives0, q=0 matches `m*(1-exp(-p*t))`, t>=0 preserves [0,m], invalid parameters fail; compare independent high-precision values.
Command: `cargo test --locked -p innovate-core --test bass_reference` after creating/locking the workspace with the qualified toolchain.
- [ ] Extract one tested slice, keep pure native execution, record red/green and review/commit; do not claim all families migrated.

### R02 — Typed observation and parameter contracts
Dependencies: R01. Create `crates/innovate-core/src/domain.rs`, `crates/innovate-core/tests/domain_contract.rs`.
Interface: `TimeGrid::try_from(Vec<f64>)`, `ObservationSeries::try_new(TimeGrid,Vec<f64>,ObservationKind,Units,Option<Denominator>)`; kinds `NewAdoption,ActiveStock,Transactions,Rate,Share,Attention`.
Test: reject NaN, unsorted/duplicate time, length mismatch, negative counts, share outside[0,1], incompatible denominator; distinguish missing from zero; observations do not infer a patient/actor count.
Command: `cargo test --locked -p innovate-core --test domain_contract`.
- [ ] Run red/green cycle, define typed error codes, review callers and commit.

### R03 — Native Polars ingestion with explicit bindings
Dependencies: R02, R08. Create `crates/innovate-data/{Cargo.toml,src/lib.rs,tests/panel_contract.rs}`.
Interface: `bind_panel(frame:polars::prelude::DataFrame,binding:&ColumnBinding)->Result<ObservationPanel,DataError>`; ColumnBinding names time/product/measure/unit/value and optional denominator explicitly.
Test: column permutations preserve output; missing required columns, wrong types, duplicate keys and ambiguous units fail; nulls remain null; lazy/eager qualified paths agree. No automatic pandas/xarray conversion.
Command: `cargo test --locked -p innovate-data --test panel_contract`.
- [ ] Create red tests, implement Polars boundary, verify streaming/resource limits on a bounded fixture, review and commit.

### R04 — Arrow and joint-draw schema
Dependencies: R02, R08. Create `crates/innovate-arrow/{Cargo.toml,src/lib.rs,tests/joint_draw_roundtrip.rs}` and `specs/joint-draw-v1.json`.
Interface: `encode_joint_draws(&JointDrawBundle)->Result<Vec<u8>,SchemaError>` and `decode_joint_draws(&[u8])->Result<JointDrawBundle,SchemaError>` using Arrow IPC.
Test: retain draw/chain/scenario/strategy/product/region/time IDs, dependence, units, fit status and provenance; reject unsupported schema, duplicate keys, missing units and invalid fit. Preserve dictionaries/nulls and buffer ownership.
Command: `cargo test --locked -p innovate-arrow --test joint_draw_roundtrip`.
- [ ] Specify schema with failing tests, implement round trips, review lifetime safety and commit.

### R05 — Direct Python native dispatch
Dependencies: R01, R03, R04, I01. Create `bindings/python/{Cargo.toml,src/lib.rs}`, `tests/installed/test_native_dispatch.py`; modify `pyproject.toml`, `src/innovate/kernel.py` and packaging.
Interface: `innovate.fit_model`/`predict_model` marshal typed requests to Rust and return Polars/Arrow-compatible results; runtime receipt identifies native artifact and algorithm.
Test: missing extension raises NativeBackendUnavailable; monkeypatching Python reference mathematics cannot change native results; subprocess interception proves no Python bridge dispatch; installed wheel works outside checkout.
Command: `uv run pytest tests/installed/test_native_dispatch.py -q` against an installed candidate, not source injection.
- [ ] Red/green tests, bind accepted slices only, reject unported operations, review and commit.

### R06 — Remove supported pandas/xarray and host-engine paths
Dependencies: R05. Modify `pyproject.toml`, `src/innovate/arrow_interchange.py`, `src/innovate/dataframe_engines.py`, relevant runtime imports; create `tests/installed/test_supported_dependency_closure.py` and migration documentation.
Interface: public input types are native/Polars/Arrow, with explicit unsupported-input error for legacy frames. External reference environments use separate locks.
Test: clean minimal and every supported-extra install has no pandas/xarray packages; runtime source/API scan has no imports or auto-conversion; deliberately supplied legacy objects fail with migration guidance; no Mesa/JAX/NumPy scientific fallback.
Command: `uv run pytest tests/installed/test_supported_dependency_closure.py -q` using generated clean envs and complete declared-extra combinations.
- [ ] Red/green retirement and closure tests, remove misleading examples, review transitive plotting/diagnostics dependencies and commit. Do not change old published artefacts.

### R07 — Native CLI and Python-free consumer
Dependencies: R04, A01. Create `crates/innovate-cli/{Cargo.toml,src/main.rs}`, `tests/installed/test_python_absent.py`.
Interface: `innovate-native fit --request FILE --output FILE` and `predict` accept versioned requests and write result/receipt without Python.
Test: container/PATH without Python executes supported operations; unsupported operation exits nonzero with structured error; malformed input does not panic; round-trip result matches library reference.
Command: `uv run pytest tests/installed/test_python_absent.py -q` orchestrates a genuinely Python-free child environment.
- [ ] Build test before implementation, prove absence inside child, review actual artifact identity and commit.

### R08 — Qualify common dependency closure
Dependencies: none. Create `conductor/qualification/dependency-candidates.json` and `conductor/qualification/dependency-receipts/`; modify root/binding manifests only with integrator ownership.
Interface: candidate/blocked/qualified records conform to repository-standards five-check schema; exact revisions and root patches are explicit.
Test: security/build/numerical/consumer/resource must all pass; removing the required Kairos Polars patch is detected; unsupported Git dependency cannot be called package-index-ready.
Commands: provider policy CLI `python scripts/scientific_ecosystem.py candidate CANDIDATE.json` in repository-standards; relevant locked native/installed tests in Innovate.
- [ ] Inspect latest candidates, record expected failed qualification first, resolve or record bounded exceptions, rerun actual gates, review and commit exact pins. Metadata validation alone cannot qualify a dependency.

### R09 — Direct native FFI and existing binding migration
Dependencies: R04, R07. Create `bindings/ffi/{Cargo.toml,src/lib.rs,tests/ffi_lifecycle.rs}`; update each existing `bindings/{r,julia,go,csharp,typescript}` adapter within its published scope.
Interface: versioned opaque handles and Arrow C Data release ownership; no caller receives Rust struct layout as stable ABI.
Test: double release, error cleanup, null/invalid handle, unsupported schema and repeated calls; supported bindings invoke the same native algorithms and cannot call Python bridge scripts.
Command: `cargo test --locked -p innovate-ffi --test ffi_lifecycle`; run each existing language's installed contract suite separately and record unsupported surfaces.
- [ ] Red/green each language slice independently, review ABI/ownership, commit; do not claim all-language parity from one smoke test.

### R10 — Executable capability matrix
Dependencies: R05, R07. Create `conductor/capabilities.json`, `scripts/validate_capability_claims.py`, `tests/unit/test_capability_claims.py`.
Interface: one record per model-operation-payload-backend with implementation/native/scientific/empirical/release evidence separately identified.
Test: absent, stale or wrong-revision native receipt blocks native claim; documentation completion cannot promote scientific validity; unknown capability is unavailable.
Command: `uv run pytest tests/unit/test_capability_claims.py -q`; `uv run python scripts/validate_capability_claims.py conductor/capabilities.json`.
- [ ] Test seeded false claims first, generate human-readable status from accepted records, review and commit without hardcoded green summaries.

## Innovation models — advanced_diffusion_frontier_20261107

### A01 — Native classical fit/predict
Dependencies: R01, R02, R08, I03. Create `crates/innovate-core/src/models/{bass,logistic,gompertz,fisher_pry}.rs`, `crates/innovate-core/tests/classical_fit.rs`.
Interface: `fit_classical(request:&FitRequest)->FitOutcome` and `predict_classical(request:&PredictionRequest)->Result<PredictionBundle,InnovationError>`; explicit family and observation type.
Test: noiseless recoverable fixtures and independent reference predictions; weakly identified pre-inflection fixtures flag uncertainty rather than fake certainty; bounds respected; failures explicit.
Command: `cargo test --locked -p innovate-core --test classical_fit`.
- [ ] Add one family red/green slice at a time, verify independently and commit; freeze optimizer tolerances before benchmark comparison.

### A02 — Count and continuous likelihoods
Dependencies: A01. Create `crates/innovate-core/src/likelihood.rs`, `crates/innovate-core/tests/likelihood_reference.rs`.
Interface: `log_likelihood(obs:&ObservationSeries,mean:&[f64],config:&LikelihoodConfig)->Result<f64,InnovationError>` with Poisson, negative-binomial and Gaussian/Student-t declared parametrisations.
Test: independent probability/log-likelihood values; zero counts; Poisson limit where appropriate; invalid dispersion; stable large-count calculations; cumulative counts not treated as independent increments.
Command: `cargo test --locked -p innovate-core --test likelihood_reference`.
- [ ] Red/green likelihood and gradient tests, verify parametrisation in exported metadata, review and commit.

### A03 — Hierarchical partial pooling
Dependencies: A02, I03. Create `crates/innovate-core/src/models/hierarchical.rs`, `crates/innovate-core/tests/hierarchical_recovery.rs`.
Interface: `fit_hierarchical(&PanelFitRequest)->FitOutcome` with group IDs, non-centred group effects and explicit hyperparameters.
Test: identical-group/no-pooling limits, known-group recovery, sparse groups shrink appropriately under the declared prior, unseen group predictive draws include hyperparameter uncertainty.
Command: `cargo test --locked -p innovate-core --test hierarchical_recovery`; add this family to I04 calibration protocol.
- [ ] Red/green implementation, compare isolated independent inference oracle and held-out groups, review and commit.

### A04 — Multi-generation substitution and competing products
Dependencies: A02. Create `crates/innovate-core/src/models/{norton_bass,competition}.rs`, `crates/innovate-core/tests/competition_conservation.rs`.
Interface: `competing_rates(t:f64,state:&MarketState,params:&CompetitionParameters)->Result<MarketRates,InnovationError>` with entry times, shared market boundary and generation transfers explicit.
Test: no-competition limit reproduces single product; product relabelling permutes outputs; no adoption before entry; feasible states remain nonnegative; transfer conserves specified market accounting; IBM/DIMORA reference cases agree.
Command: `cargo test --locked -p innovate-core --test competition_conservation`.
- [ ] One model slice per commit; reject unsupported identifiability/observation combinations; review numerical stability.

### A05 — Repeat use, switching and abandonment
Dependencies: A02. Create `crates/innovate-core/src/models/repeat_use.rs`, `crates/innovate-core/tests/repeat_use_balance.rs`.
Interface: `step_active_use(&ActiveUseState,&UseRates,dt:f64)->Result<ActiveUseState,InnovationError>` and `expected_transactions(&ActiveUseState,&SupplyIntensity)->Result<Vec<f64>,InnovationError>`.
Test: starts plus in-switches minus stops/out-switches balance stock; transaction counts are not patient counts; zero-repeat limit and steady-state analytic fixtures; non-identifiable aggregate scenarios produce diagnostics/bounds.
Command: `cargo test --locked -p innovate-core --test repeat_use_balance`.
- [ ] Red/green generic capability, test a non-medical repeat-use fixture as well as medicines, review and commit.

### A06 — Regimes, shocks and changing market potential
Dependencies: A02. Create `crates/innovate-core/src/models/regimes.rs`, `crates/innovate-core/tests/regime_boundaries.rs`.
Interface: `predict_regimes(&RegimeRequest)->Result<PredictionBundle,InnovationError>` with dated forcing and change-point semantics.
Test: no-shock limit matches baseline; split integration at discontinuities; no future intervention used before known date; decreasing capacity is not silently represented as negative cumulative adopters.
Command: `cargo test --locked -p innovate-core --test regime_boundaries`.
- [ ] Red/green with known changes and no-change negatives; compare smooth alternatives, review and commit.

### A07 — Networks, complements and path dependence
Dependencies: A04, E01. Create `crates/innovate-core/src/models/network.rs`, `crates/innovate-kairos/tests/network_limits.rs`.
Interface: `InnovationBehaviour` consumes explicit topology/exposure and emits adoption/cessation actions into Kairos; model parameters remain Innovate-owned.
Test: no-edge/no-complement limits; node-permutation invariance; topology changes and exposure history; hysteresis sweep compared with independent small-state solution; one clock/RNG authority.
Command: `cargo test --locked -p innovate-kairos --test network_limits`.
- [ ] Red/green limited topology slice, then expand with Microfinance/temporal-network evaluation; do not equate an exposure coefficient with a causal spillover.

### A08 — Structural demand and innovation-specific strategic models
Dependencies: A04, G02. Create `crates/innovate-core/src/models/demand.rs`, `crates/innovate-games/src/market_game.rs`, `crates/innovate-games/tests/demand_game_reference.rs`.
Interface: `choice_shares(&ChoiceSet,&PreferenceParameters)->Result<Vec<f64>,InnovationError>` and `build_launch_game(&InnovationMarket)->Result<GameSpec,InnovationError>`.
Test: outside option and shares sum correctly; independent logit elasticities; dominated launch strategies removed only when assumptions justify; tiny timing-game equilibrium agrees with analytic solution. Prices are endogenous unless an explicit identification design says otherwise.
Command: `cargo test --locked -p innovate-games --test demand_game_reference`.
- [ ] Start with logit/static/two-period cases; register BLP, real options and dynamic oligopoly as separately specified extensions with code-availability and identification audits, not one opaque solver task.

## Hype — healthcare_hype_validation_20261107

### H01 — Correct composite shape and derivatives
Dependencies: R02. Create `crates/innovate-core/src/models/hype.rs`, `crates/innovate-core/tests/hype_shape.rs`.
Interface: `hype_mean(t:f64,p:&HypeParameters)->Result<f64,InnovationError>` and `hype_derivative(...)`; parameters include baseline, free maturity amplitude, growth, timing and pulse widths/amplitudes.
Test: no-pulse limit is maturity; derivative matches independent finite difference; zero/negative widths and NaN rejected; combined extrema solve derivative of the sum; no hard clipping of latent expectation. Distinct ordered-cycle and unrestricted-wave configurations.
Command: `cargo test --locked -p innovate-core --test hype_shape`.
- [ ] Red/green tests before changing legacy visual output; verify smoothness/limits, review and commit.

### H02 — Joint composite likelihood fitting
Dependencies: H01, A02. Create `crates/innovate-core/src/inference/hype_fit.rs`, `crates/innovate-core/tests/hype_fit.rs`.
Interface: `fit_hype(&HypeFitRequest)->FitOutcome`, returning all component parameters, uncertainty/identifiability diagnostics and fitted-total predictions; training-only time normalisation is stored.
Test: recover noiseless identifiable parameters from several starting points; recognise indistinguishable decompositions; predictive total can be identified when components are not. Optional variable projection only for identity-Gaussian case; count link uses full joint likelihood.
Command: `cargo test --locked -p innovate-core --test hype_fit`.
- [ ] Implement constrained multistart baseline with red/green recovery tests, compare independent least-squares oracle, review and commit.

### H03 — Native Bayesian composite and partial pooling
Dependencies: H02, A03, I04. Create `crates/innovate-core/src/inference/hype_posterior.rs`, `crates/innovate-core/tests/hype_posterior.rs`.
Interface: `sample_hype_posterior(&HypePosteriorRequest)->FitOutcome` using qualified native sampler/log-density gradients and optional group effects.
Test: ordered parameter transforms/Jacobians are correct; weak/no-cycle prior-predictive cases allowed; non-convergence/invalid sampler state rejects posterior promotion; SBC and repeated coverage retain all failures.
Command: `cargo test --locked -p innovate-core --test hype_posterior`; execute I04 protocol with this model.
- [ ] Red/green contracts, native sampler qualification and calibration before claiming Bayesian validity; review priors/identifiability and commit.

### H04 — Separate hype effects on adoption and use
Dependencies: H02, A01, A05. Create `crates/innovate-core/src/models/hype_effects.rs`, `crates/innovate-core/tests/hype_effects.rs`.
Interface: `innovation_rates(t:f64,state:&InnovationState,hype:&HypeSignal,p:&HypeEffectParameters)->Result<InnovationRates,InnovationError>` with separate log-p/log-q coefficients and optional cessation hazard.
Test: zero coefficients recover baseline; negative coefficient can inhibit uptake; cumulative first adoption cannot fall; cessation changes active stock; separately supplied effects do not silently become a shared time multiplier.
Command: `cargo test --locked -p innovate-core --test hype_effects`.
- [ ] Red/green separate mechanisms, state assumptions and recoverability, review and commit.

### H05 — Delay semantics and endogenous feedback
Dependencies: H04, A06, E01. Create `crates/innovate-core/src/models/expectation_feedback.rs`, `crates/innovate-core/tests/expectation_feedback.rs`; audit legacy `src/innovate/hype/delayed_hype_bass.py`.
Interface: separate `ExternalLagSignal` from `DelayedStateModel`; joint feedback uses explicit hype and adoption state equations and measurement models.
Test: zero lag reduces to baseline; shifted external forcing agrees with a non-DDE reference; genuine state delay requires history; no-effect/reverse/shared-shock examples expose identification limits.
Command: `cargo test --locked -p innovate-core --test expectation_feedback`.
- [ ] Audit equation first, red/green one mechanism at a time; retain experimental label without sufficient data, review and commit.

### H06 — Hype impact benchmark
Dependencies: H02, H04, M05, V02. Create `validation/hype/protocol.json`, `validation/hype/run.py`, `tests/validation/test_hype_protocol.py`.
Interface: frozen model/data manifest includes no-cycle, single/repeated-wave, flexible spline/state-space and no-hype adoption comparators; native fit outputs consumed by orchestration.
Test: selection never sees held-out technology/time; no-peak synthetic data is not forced into a cycle; effects and curve fit scored separately; attention proxies labelled; failure and null results retained.
Command: `uv run pytest tests/validation/test_hype_protocol.py -q`, then versioned native benchmark jobs.
- [ ] Freeze and test protocol, run comparative evaluation, review incremental scores/calibration and commit report; no claim-exaggeration annotation prerequisite.

## Medicines and datasets — empirical_validation_publication_20261107

### M01 — Exact existing source manifest
Dependencies: none. Create `validation/data/pbs-sources.json`, `tests/validation/test_pbs_manifest.py`; consume the six paths/hashes in `research/medicines-utilisation-20261107.md`.
Interface: manifest identifies HF revision, object digest, upstream receipt, rights/sensitivity scope and separate byte/semantic status. No new harvester.
Test: six unique exact objects; changed byte/hash or wrong revision rejected; later rights assessment not overwritten by historical unknown labels; absent modelling qualification remains absent.
Command: `uv run pytest tests/validation/test_pbs_manifest.py -q` using local receipt fixtures, not unrestricted network in fast CI.
- [ ] Red/green manifest checks, verify actual permitted acquisition in a separately logged job, review and commit.

### M02 — Source-faithful Polars projection
Dependencies: M01, R03. Create `crates/innovate-data/src/atlas_import.rs`, `crates/innovate-data/tests/atlas_import.rs`; source parsing/qualification changes, if needed, are upstream Atlas handoffs.
Interface: `read_atlas_projection(path:&Path,manifest:&SourceManifest,binding:&ColumnBinding)->Result<ObservationPanel,DataError>` accepts qualified Arrow/Parquet or explicit native CSV parsing; workbook conversion uses a qualified Rust reader upstream.
Test: field/dtype/units preserved; duplicate source records and invalid date basis rejected; absent optional fields remain null; no pandas-based Excel path or invented worksheet schema.
Command: `cargo test --locked -p innovate-data --test atlas_import`.
- [ ] Red/green bounded fixture, upstream canary/semantics receipts, review and commit; file names alone never set coverage.

### M03 — Temporal mapping and panel qualification
Dependencies: M02. Create `crates/innovate-data/src/utilisation_panel.rs`, `crates/innovate-data/tests/utilisation_panel.rs`.
Interface: `qualify_utilisation(&ObservationPanel,&MappingHistory)->Result<QualifiedPanel,QualificationError>`.
Test: mapping effective dates obey as-of information; unmapped codes preserved; summary/detailed overlap is not double counted; scripts/stock/spend remain distinct; changed packs/codes and missing months flagged; no implicit unique-patient interpretation.
Command: `cargo test --locked -p innovate-data --test utilisation_panel`.
- [ ] Red/green tests, generate coverage/grain report from actual values, review and commit before modelling.

### M04 — Generic utilisation observation model
Dependencies: M03, A05. Create `crates/innovate-core/tests/utilisation_identifiability.rs` and `examples/medicines/utilisation-model.json`.
Interface: application binds active-use model, supply intensity, coverage and source measures; generic model remains A05-owned.
Test: independently generated starts/continuation/switches recover under adequate measurements; two latent histories with equal aggregate scripts trigger non-identifiability; quantities cannot become utility or patient counts silently.
Command: `cargo test --locked -p innovate-core --test utilisation_identifiability`.
- [ ] Red/green identification/bounds tests, add non-medical repeat-use example, review and commit.

### M05 — Frozen medicines forecast cohort
Dependencies: M03, A01, V02. Create `validation/medicines/protocol.json`, `validation/medicines/run.py`, `tests/validation/test_medicines_protocol.py`.
Interface: manifest fixes observed coverage, cohorts, code mappings, training origins, horizons, source vintages and price/quantity outcomes separately.
Test: training-only saturation/scaling; final holdout never tunes models; annual/monthly overlaps handled; sparse/new/zero series retained with explicit eligibility; insufficient history fails the proposed 24-month-training/12-month-holdout design rather than padding data.
Command: `uv run pytest tests/validation/test_medicines_protocol.py -q`.
- [ ] Freeze after source pilot, run named baselines and native candidates, review paired uncertainty/failures and commit full results.

## Game-runtime boundary — advanced_diffusion_frontier_20261107

### G01 — Inventory and consolidate generic game ownership
Dependencies: none. Provider: `new-drug-reimbursement-game`, coordinate Kairos track110 S01–S04. Inspect `crates/uogto-game-core`, `crates/uogto-game-solve`, and Kairos `crates/kairo-ecs-game-{theory,ontology}`; create provider `docs/architecture/game-runtime-extraction.md` and duplicate-owner tests.
Interface: one authoritative GameSpec/Policy/Solution contract; UOGTO is semantic validation; Kairos owns execution only; no second scheduler.
Test: static solver dependency graph contains no application or mandatory simulator; each existing advertised algorithm maps to one owner and an accepted/unsupported/migration state; no completion obligation disappears.
Command: provider `cargo test --workspace --locked` plus a checked-in ownership-graph test introduced with this task.
- [ ] Inventory callers before moving code; document retained adapters and exact extraction boundary; review and commit. A new repository is not evidence of extraction.

### G02 — Independently consumable solver slice
Dependencies: G01, R08. Provider files: `crates/uogto-game-core`, `crates/uogto-game-solve`; create `crates/uogto-game-solve/tests/solver_oracles.rs` and installed consumer fixtures.
Interface: `solve_finite_game(&GameSpec,&SolveConfig)->Result<Solution,SolveError>` reports algorithm, supported scope, residual/exploitability and selection among multiple equilibria.
Test: analytic backward induction, strict dominance, matching pennies, coordination/multiple equilibria, rectangular and unsupported games; CE incentive inequalities checked independently; algorithm-specific unsupported inputs fail. Complete one algorithm per reviewed slice.
Command: provider `cargo test --locked -p uogto-game-solve --test solver_oracles`.
- [ ] Use isolated Gambit/OpenSpiel reference outputs, not production dependency; qualify reimbursement and Innovate installed consumers; review and commit before generic-library promotion.

### G03 — Optional dynamic execution adapter
Dependencies: G02, E01. Provider creates `crates/uogto-game-kairos/{Cargo.toml,src/lib.rs,tests/replay.rs}`; Innovate creates `crates/innovate-games/{Cargo.toml,src/lib.rs}`.
Interface: game actions/observations dispatch through Kairos public APIs; stateless solving is independent, repeated-game world state lives in Kairos.
Test: no duplicate clock/RNG; same seed/action history reproduces trajectory; disabled adapter does not build Kairos; unsupported information sets rejected rather than mapped to a special-case game.
Command: provider `cargo test --locked -p uogto-game-kairos --test replay`; Innovate installed-adapter smoke.
- [ ] Red/green adapter, record immutable provider pin and public API, review and commit.

### G04 — Strategic innovation research cases
Dependencies: A08, G03, V02. Create `validation/competition/protocol.json`, `validation/competition/run.py`, `tests/validation/test_competition_protocol.py`.
Interface: prespecified cases for launch timing, price/quality competition, switching costs, cannibalisation and optional investment/learning; distinguish equilibrium selection assumptions.
Test: replicate a named published/analytic special case before extension; payoff/price transformations preserve stated invariants; counterfactual differences include uncertainty and multiple-equilibrium sensitivity.
Command: `uv run pytest tests/validation/test_competition_protocol.py -q` then native solver/model jobs.
- [ ] Audit literature and available code/licence per case; no claim an established method lacks software until documented; review and commit comparative results.

## Ecosystem integration — ecosystem_decision_integration_20261107

### E01 — Real Kairos execution
Dependencies: R02, R08. Create `crates/innovate-kairos/{Cargo.toml,src/lib.rs,tests/native_execution.rs}`; retire `src/innovate/abm/kairos_adapter.py` reference fallback from supported dispatch.
Interface: `run_innovation(&SimulationRequest,&KairosRuntime)->Result<SimulationResult,InnovationError>` uses accepted public Kairos event/state/RNG APIs; Innovate supplies behaviours.
Test: native engine identity is present; reference Python function is never called; missing provider fails; deterministic replay and resource conservation pass; no second clock.
Command: `cargo test --locked -p innovate-kairos --test native_execution`.
- [ ] Red/green actual runtime test, qualified pin and installed consumer, review and commit.

### E02 — Uncertainty-preserving Voiage handoff
Dependencies: R04, A01. Create `specs/innovation-decision-binding-v1.json`, `tests/integration/test_voiage_handoff.py`; provider adapter belongs under Voiage optional ecosystem integration.
Interface: `DecisionBinding` explicitly maps joint prediction/parameter draws to strategy-labelled utilities/net benefits, perspective, population, horizon and discount basis.
Test: correlated draw IDs survive; swapped labels/units/invalid fits are rejected; a two-strategy finite fixture agrees with independent EVPI calculation; prescription counts without utility mapping cannot become Voiage values.
Command: `uv run pytest tests/integration/test_voiage_handoff.py -q` against installed provider/candidate packages.
- [ ] Red/green contract and installed adapter, record provider PR/pin, review and commit. Do not hard-depend Innovate core on Voiage.

### E03 — Decision feedback without build cycles
Dependencies: E02. Create `specs/study-design-v1.json`, `tests/integration/test_study_design_feedback.py` and an application-level orchestration example.
Interface: Voiage-produced `StudyDesign`/`PolicyScenario` artefacts return via orchestration to Innovate; cores share versioned data, not mutual imports.
Test: dependency graph stays acyclic; unsupported design rejected; observations simulated from the declared study model; same design/seed reproduces output; no generic variance reduction substituted for EVSI.
Command: `uv run pytest tests/integration/test_study_design_feedback.py -q`.
- [ ] Red/green finite design, compare independent small decision oracle, review and commit.

### E04 — Promote VOP proof-of-concept capabilities to Voiage
Dependencies: E02. Provider: Voiage. Inspect `vop_poc_nz` exported CEA/DCEA/VOI/perspective/Sobol helpers and `voiage/rust/crates/{voiage-domain,voiage-numerics}`; create provider extraction ledger and `rust/crates/voiage-numerics/tests/vop_reference.rs`.
Interface: generic perspective/equity/decision and information-value operations are Voiage-owned; NZ scenarios/parameters remain in the POC. Simulation/RNG belongs Kairos where needed, adoption belongs Innovate.
Test: independently computed small perspective/regret/equity fixtures; unchanged empirical scenario output within justified tolerance; POC calls library rather than copying kernels. No assertion all POC functions are correct before audit.
Command in Voiage: `cargo test --manifest-path rust/Cargo.toml --locked -p voiage-numerics --test vop_reference`; installed POC consumer test.
- [ ] Inventory and test each extracted method, retire duplicate public framework claim, record migration receipts, review and commit.

### E05 — Application promotion and surrogate gate
Dependencies: E01, E02, M04. Create `conductor/qualification/application-promotion.json`, `tests/integration/test_application_contracts.py`; coordinate tobac/careops-sim/MARS without copying their domain code.
Interface: ledger maps demonstrated reusable method to provider, domain residue, independent oracle and installed consumer. MARS only approximates declared expensive model functions.
Test: two application consumers share the same provider method; no library imports an application; surrogate agrees on held-out simulations and preserves decision ranking within registered tolerance; out-of-domain use fails/abstains.
Command: `uv run pytest tests/integration/test_application_contracts.py -q` plus provider-native tests.
- [ ] Red/green handoffs, include negative surrogate decisions and resource costs, review and commit.

## Validation and publication — empirical_validation_publication_20261107

### V01 — Independent model-oracle registry
Dependencies: R02. Create `validation/oracles/manifest.json`, `tests/validation/test_oracle_manifest.py` and fixtures for classical/substitution/competition models.
Interface: each oracle records derivation or independent package/version, licence, input hash, expected quantities and tolerances. No fixtures generated solely by the kernel under test.
Test: identical implementation hash cannot be labelled independent; stale inputs/unknown tolerance fail; analytic limits and known wrong outputs are caught.
Command: `uv run pytest tests/validation/test_oracle_manifest.py -q`.
- [ ] Red/green registry, acquire reference evidence in isolated env, review and commit.

### V02 — Leakage-safe benchmark runner
Dependencies: R04, V01. Modify `src/innovate/benchmarks/runner.py`, `src/innovate/benchmarks/corpus.py`; create `tests/validation/test_temporal_benchmark.py`.
Interface: `BenchmarkProtocol` fixes source vintages, origins, horizons, tuning budget, native candidate and isolated baseline references; runner orchestrates only.
Test: future scaling/saturation/revised data and duplicate holdouts rejected; failed fits stay in denominators; stochastic seeds recorded; output includes point/probabilistic scores, calibration, timing and memory.
Command: `uv run pytest tests/validation/test_temporal_benchmark.py -q`.
- [ ] Red/green adversarial leakage tests, review and commit before examining final holdout results.

### V03 — Cross-domain and early-stage validation
Dependencies: V02, A01. Create `validation/cross-domain/protocol.json`, `tests/validation/test_cross_domain_protocol.py`.
Interface: qualified CHAT/CER, IBM/DIMORA and policy/network cohorts map to their actual observation semantics; source rights/version/coverage explicit.
Test: leave-technology/region-out splits and pre-inflection windows; publication-count/usage measures not relabelled new adopters; weak market-potential identification reported.
Command: `uv run pytest tests/validation/test_cross_domain_protocol.py -q`; run frozen native/reference jobs.
- [ ] Freeze data/protocol, compare strong simple baselines and hierarchical models, report calibration/failures/negative findings, review and commit.

### V04 — Fair performance and integration experiments
Dependencies: R07, E01, V02. Create `validation/performance/protocol.json`, `crates/innovate-core/benches/workloads.rs`, `tests/validation/test_performance_receipts.py`.
Interface: equivalent numerical workloads with cold/warm time, compilation, transfer, peak memory, hardware and accuracy recorded; optional ensembles/parallel/GPU separately identified.
Test: fake device/metadata-only execution rejected; different work/precision cannot be called a speedup; preserve genuine slowdowns and crossover results.
Command: `cargo bench --locked -p innovate-core --bench workloads`; `uv run pytest tests/validation/test_performance_receipts.py -q`.
- [ ] Measure sequential first, optimise only observed bottlenecks, review correctness and receipt integrity, commit.

### V05 — Reproduction and publication packages
Dependencies: V03, M05, Q03. Modify `paper.qmd`, `references.bib`, `CITATION.cff`; create paper-specific immutable manifests and `tests/validation/test_reproduction_bundle.py`.
Interface: a clean-user command reproduces declared tables/figures from rights-permitted pinned inputs and installed native artefacts.
Test: missing/changed input fails; reported values match computation; no private source redistributed; AI assistance and exact software/data versions disclosed; unrelated studies are not duplicate publications.
Command: `uv run pytest tests/validation/test_reproduction_bundle.py -q` and the recorded clean-environment reproduction command.
- [ ] Separate JOSS software, forecasting, innovation/hype and decision/data outputs; verify current venue rules; review and commit. External submission remains a separately authorised action.

## Hardening and release — empirical_validation_publication_20261107

### Q01 — Malformed-input, property and FFI safety gates
Dependencies: R04, R09. Create `fuzz/`, `crates/innovate-core/tests/properties.rs`, `bindings/ffi/tests/adversarial.rs` and bounded CI smoke targets.
Interface: all native boundaries return typed errors on malformed inputs; no panics across FFI; deterministic state ownership.
Test: overflow, oversized payload, invalid UTF-8/schema, null/duplicate keys, release-after-error, cancellation and interrupted files; seeded corpus replay; conservation/property invariants.
Command: `cargo test --workspace --locked`; named bounded fuzz runs documented with corpus/hash and stop conditions.
- [ ] Red/green regressions for discovered faults, run sanitizers where supported, review safety and commit without universal memory-safety claims from one run.

### Q02 — Native and no-legacy CI matrix
Dependencies: R06, R07, R09. Modify `.github/workflows/ci.yml`, binding/package workflows; create `tests/installed/test_release_matrix.py`.
Interface: declared platform/architecture/native artifact matrix with actual installation and method execution, not only imports.
Test: Windows x64/Linux x64/macOS arm64 where advertised; missing native artifact and pandas/xarray reintroduction fail; one unsupported lane cannot be marked passed/skipped as support proof.
Command: `uv run pytest tests/installed/test_release_matrix.py -q`; hosted matrix results bound to exact head SHA.
- [ ] Red/green matrix assertions, preserve stronger existing gates, review workflow security and commit.

### Q03 — Supply chain and reproducible artifacts
Dependencies: R08, Q02. Modify release workflows/packaging; create `conductor/qualification/release-artifacts.json` and `tests/installed/test_artifact_provenance.py`.
Interface: artifact hashes, lock/toolchain/source IDs, SBOM/licence/security records and actual publishability status per package.
Test: tampered artifact/hash, unapproved licence, missing native library or unpublishable Git closure blocks candidate; clean wheel/crate consumers reproduce declared examples.
Command: `uv run pytest tests/installed/test_artifact_provenance.py -q`; locked Cargo audit/deny and packaging tools only after their exact versions are recorded.
- [ ] Red/green corruption/packaging tests, inspect actual built artifacts, review and commit; no publication in this task.

### Q04 — Scientific and documentation release truth
Dependencies: R10, I04, V03, Q01. Create `tests/validation/test_release_claims.py`; generate capability/support docs from evidence.
Interface: release manifest enumerates stable/experimental/unavailable model-operation-payload combinations and outstanding completion obligations.
Test: changed algorithm or stale receipt revokes acceptance; fake posterior/identity-inference/wrong-method mutations caught; unsupported hype/network/game scope not claimed stable; examples execute installed packages.
Command: `uv run pytest tests/validation/test_release_claims.py -q`; coverage/mutation checks per execution contract.
- [ ] Red/green seeded false-claim cases, review docs/migration warnings, commit without deleting unresolved requirements to make status green.

### Q05 — Beta, RC and stable acceptance
Dependencies: Q02, Q03, Q04, V04, E02. Modify `conductor/qualification/release-decision.json`, release checklist and changelog.
Interface: decision records exact candidate, all mandatory gate receipts, declared supported scope, blockers and publication authorisation state.
Test: any fail/skip/unrun/expired mandatory gate blocks promotion; a historical green build cannot qualify a new source; current source is not called full-native until all advertised stable combinations prove it.
Command: existing release-readiness workflow plus actual installed artefact matrix on the exact release commit.
- [ ] Execute beta use/reproduction, RC hardening and stable decision in order; retain unresolved experimental scope honestly; review and commit. Publish only through a separately authorised release action.
