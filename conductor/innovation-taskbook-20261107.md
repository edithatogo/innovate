# Innovation Native Capability Implementation Plan

> For agentic workers: execute one ready task at a time using superpowers:executing-plans or superpowers:subagent-driven-development. Every task remains open. Read the owning track spec and this task's dependencies before editing.

**Goal:** Deliver a general innovation-modelling library with a mandatory Rust core, Polars/Arrow interfaces, validated scientific methods and a hardened scoped release.

**Architecture:** Native core plus data/interchange and optional Kairos/game adapters; Voiage consumes joint outputs through an optional integration boundary. Applications own their scenarios.

**Tech Stack:** Qualified Rust numerical dependencies, Polars/Arrow, native bindings, Kairos, isolated independent test oracles and versioned benchmark data.

**Spec:** `product.md`, `frontier-programme-20261107.md`, `execution-contract-20261107.md`, the owning track's `spec.md`, and linked research protocols. The 8 November scientific refinements below amend the existing 52 packets; they do not add a parallel completion ledger.

## Global constraints

Rust owns scientific execution; no production Python fallback. Pandas/xarray are unsupported in the new release, including transitive supported extras. Root pins/shared types have one integrator. Current hybrid code is not retrospectively certified. Proposed paths and commands below become runnable when their prerequisite task creates the corresponding target. External oracle environments are separate from production installs. No task closes from a document alone.

Paignia's authoritative repository is `edithatogo/paignia`; preserve transferred T17 IDs and incubator provenance. Native core, data/bindings and optional integrations qualify independently. An unfinished optional integration does not block an unrelated scoped release, but every claimed capability keeps all applicable evidence and unfinished obligations remain visible. Do not infer package availability from repository existence.

## Review focus

Wrong stock/flow/transaction meanings (R02/M03/M04); invalid or stale posterior state and prior-only inference (I01/I03/I04); secretly falling back to Python (R05/R06/E01); composite and scale non-identifiability (H01-H06); cyclic dependencies, hidden information or simulation noise treated as learnable uncertainty (G02/G03/E02/E03). Each has named tests below.

## How to execute each packet

For every task: (1) create the named test with the stated assertions; (2) run the listed command and record the expected red result; (3) implement the stated interface in the listed files; (4) rerun that command plus affected contracts and record green output; (5) review, commit and attach a receipt. All five are mandatory. Commands using new test targets require creating those targets first; compilation failure alone does not substitute for a behavioural regression after an interface exists. Cross-repository tasks run in the named provider, never as copied implementation inside Innovate. Reuse current public types after inventory; signatures below are proposed contracts, not claims that they already exist.

Each implementing agent records a concrete input/expected result, prohibited shortcut and stop condition. Stop and report an unresolved scientific choice or absent source field rather than inventing an assumption. Do not split one invariant across uncoordinated implementations. A zero-test Cargo filter is not acceptance. Separate execution, convergence, identification, prediction and causal-assumption assessments; a weakly identified component may still support an explicitly bounded predictive result.

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
Interface: `FitOutcome::{Valid(FitResult),Invalid(FitFailure)}` and `posterior(&FitOutcome)->Result<&PosteriorDraws,InnovationError>`; `log_density(theta:&[f64], data:&ObservationSeries)->Result<(f64,Vec<f64>),InnovationError>`. FitResult carries separate convergence, identification, predictive-evidence and causal-assumption assessments; Valid means execution validity, not all scientific claims.
Test: invalid states cannot expose draws; analytic Gaussian log-density/gradient agree with independent formula and finite differences; zero/invalid variance is rejected. Reset does not retain prior results. A numerically valid weakly identified fit retains that warning and cannot be promoted to a causal result.
Commands: `cargo test --locked -p innovate-core --test inference_status --test gradient_oracle`.
- [ ] Create types/tests, record red, implement minimal native contract, record green, review and commit.

### I04 — Coverage and calibration harness
Dependencies: I03, A01. Create `validation/inference/protocol.json`, `validation/inference/run.py`, `tests/validation/test_inference_protocol.py`.
Interface: runner consumes immutable native executable plus simulator/seed manifest, emits parameter and data-dependent ranks, intervals, failure status and summary receipts; Python orchestrates only.
Test: injected biased estimates and fake narrow intervals are detected; failed replicates count in denominator; exact conjugate-normal fixture calibrates within prespecified Monte Carlo bounds. Add an intentionally prior-only algorithm ignoring data, omitted-observation and shuffled-group controls. Parameter ranks alone cannot accept it: include joint/selected pointwise log likelihood and posterior contraction under informative data. Predeclare false-alarm limits, replicate counts and power against these negatives.
Command: `uv run pytest tests/validation/test_inference_protocol.py -q`; execute `uv run python validation/inference/run.py --protocol validation/inference/protocol.json` only after native candidates exist.
- [ ] Run red/green harness tests, retain full results and stochastic error bounds, review and commit; non-rejection is not proof of universal calibration.

## Rust core — rust_core_convergence_20261107

### R01 — Extract a binding-independent native workspace
Dependencies: none. Modify `bindings/rust/src/lib.rs`, `bindings/rust/Cargo.toml`; create root `Cargo.toml`, `crates/innovate-core/Cargo.toml`, `crates/innovate-core/src/lib.rs`, `crates/innovate-core/tests/bass_reference.rs` and Cargo.lock.
Interface: `bass_cumulative(t:f64,p:f64,q:f64,m:f64)->Result<f64,InnovationError>`; bindings re-export rather than duplicate it.
Test: t=0 gives0, q=0 matches `m*(1-exp(-p*t))`, t>=0 preserves [0,m], invalid parameters fail; compare independent high-precision values.
Command: `cargo test --locked -p innovate-core --test bass_reference` after creating/locking the workspace with the qualified toolchain.
- [ ] Extract one tested slice, keep pure native execution, record red/green and review/commit; do not claim all families migrated.

### R02 — Typed observation and parameter contracts
Dependencies: R01. Create `crates/innovate-core/src/domain.rs`, `crates/innovate-core/tests/domain_contract.rs`.
Interface: `TimeGrid::try_from(Vec<f64>)`, `ObservationSeries::try_new(TimeGrid,Vec<f64>,ObservationKind,Units,Option<Denominator>)`; kinds `NewAdoption,ActiveStock,Transactions,Rate,Share,Attention`. Keep stable scientific types independent of Polars/Rust backend layouts.
Test: reject NaN, unsorted/duplicate time, length mismatch, negative counts, share outside[0,1], incompatible denominator; distinguish missing from zero; observations do not infer a patient/actor count. Effective time and known-at time have different meanings and may not be silently substituted.
Command: `cargo test --locked -p innovate-core --test domain_contract`.
- [ ] Run red/green cycle, define typed error codes, review callers and commit.

### R03 — Lazy native Polars ingestion with explicit materialisation
Dependencies: R02, R08. Create `crates/innovate-data/{Cargo.toml,src/lib.rs,tests/panel_contract.rs}`.
Interface: `plan_panel(frame:polars::prelude::LazyFrame,binding:&ColumnBinding)->Result<PanelPlan,DataError>`; `materialise_panel(plan:PanelPlan,budget:&ResourceBudget)->Result<ObservationPanel,DataError>`. `bind_panel(DataFrame,&ColumnBinding)` is an eager convenience over the same semantics. Bind time/product/measure/unit/value and optional denominator explicitly.
Test: column permutations preserve output; missing columns, wrong types, duplicate keys and ambiguous units fail; nulls remain null. Qualified lazy/eager/partitioned paths agree within declared numerical tolerance. Projection/filtering remain in the lazy plan until the explicit boundary; bounded memory tests must execute the workload. Unsupported streaming operators fall back only to a budgeted declared mode or fail, never silently overrun memory. No pandas/xarray conversion.
Command: `cargo test --locked -p innovate-data --test panel_contract`.
- [ ] Create red tests, implement boundary and resource checks, review query plan and measured execution, commit.

### R04 — Arrow and nested joint-draw schema
Dependencies: R02, R08. Create `crates/innovate-arrow/{Cargo.toml,src/lib.rs,tests/joint_draw_roundtrip.rs}` and `specs/joint-draw-v1.json`.
Interface: `encode_joint_draws(&JointDrawBundle)->Result<Vec<u8>,SchemaError>` and `decode_joint_draws(&[u8])->Result<JointDrawBundle,SchemaError>` using Arrow IPC. Reconcile existing versioned provider types rather than inventing a universal schema.
Test: retain parameter-draw/chain/model/equilibrium/simulation-replicate/scenario/strategy/product/region/time IDs, measurement/valuation versions, dependence, units, fit assessments and provenance. Distinguish parameter, structural, simulation, measurement, strategic and numerical uncertainty without asserting independence. Reject unsupported schema, ambiguous identity, duplicate keys, missing units and invalid execution. Preserve dictionaries/nulls and buffer ownership.
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

### R08 — Qualify common dependency and feature closure
Dependencies: none. Create `conductor/qualification/dependency-candidates.json` and `conductor/qualification/dependency-receipts/`; modify root/binding manifests only with integrator ownership.
Interface: candidate/blocked/qualified records conform to repository-standards five-check schema; exact revisions, enabled features and root patches are explicit. Candidate-testing and reproducible release locks are separate lanes.
Test: security/build/numerical/consumer/resource must all pass; removing the required Kairos Polars patch is detected; unsupported Git dependency cannot be called package-index-ready. Use standalone consumers for minimal/default/declared-extra profiles and resolved normal/build dependencies. Seed a transitive optional-feature violation: basic fitting must not pull Kairos/Paignia/Voiage, static games must not pull Kairos, and ordinary ingestion must not enable undeclared GPU/distributed support. A workspace all-features pass does not prove isolation.
Commands: provider policy CLI `python scripts/scientific_ecosystem.py candidate CANDIDATE.json` in repository-standards; locked native/installed tests and recorded `cargo tree -e features`/metadata for each standalone consumer.
- [ ] Inspect latest candidates, record expected failed qualification first, resolve or retain bounded exceptions, rerun actual gates, review and commit exact pins. Metadata validation alone cannot qualify a dependency.

### R09 — Direct native FFI and existing binding migration
Dependencies: R04, R07. Create `bindings/ffi/{Cargo.toml,src/lib.rs,tests/ffi_lifecycle.rs}`; update each existing `bindings/{r,julia,go,csharp,typescript}` adapter within its published scope.
Interface: versioned opaque handles and Arrow C Data release ownership; no caller receives Rust or third-party dataframe struct layout as stable ABI. Adapter-level dependency types have a documented compatibility range.
Test: double release, error cleanup, null/invalid handle, unsupported schema and repeated calls; supported bindings invoke the same native algorithms and cannot call Python bridge scripts. A backend upgrade must not change the stable wire/domain contract without a versioned migration.
Command: `cargo test --locked -p innovate-ffi --test ffi_lifecycle`; run each existing language's installed contract suite separately and record unsupported surfaces.
- [ ] Red/green each language slice independently, review ABI/ownership, commit; do not claim all-language parity from one smoke test.

### R10 — Executable capability matrix
Dependencies: R05, R07. Create `conductor/capabilities.json`, `scripts/validate_capability_claims.py`, `tests/unit/test_capability_claims.py`.
Interface: one record per model-operation-payload-backend with implementation/native/scientific/empirical/release evidence separately identified; generate repeated public capability summaries from this source.
Test: absent, stale or wrong-revision native receipt blocks native claim; documentation completion cannot promote scientific validity; unknown capability is unavailable. Optional release readiness cannot become a mandatory core edge through generated documentation.
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
Interface: `step_active_use(&ActiveUseState,&UseRates,dt:f64)->Result<ActiveUseState,InnovationError>` and `expected_transactions(&ActiveUseState,&SupplyIntensity)->Result<Vec<f64>,InnovationError>`; active-use transitions and recording/supply intensity are distinct.
Test: starts plus in-switches minus stops/out-switches balance stock; transaction counts are not patient counts; zero-repeat and steady-state analytic fixtures; ambiguous aggregate histories produce diagnostics/bounds. A stationary stock of 100 users with supply every one versus two months has expected monthly transactions 100 versus50 without any abandonment.
Command: `cargo test --locked -p innovate-core --test repeat_use_balance`.
- [ ] Red/green generic capability, test a non-medical repeat-use fixture as well as medicines, review and commit.

### A06 — Regimes, shocks and changing market potential
Dependencies: A02. Create `crates/innovate-core/src/models/regimes.rs`, `crates/innovate-core/tests/regime_boundaries.rs`.
Interface: `predict_regimes(&RegimeRequest)->Result<PredictionBundle,InnovationError>` with dated forcing, an explicit eligible-market schedule M(t), risk-set and change-point semantics.
Test: no-shock/constant-M limits match baseline; split integration at discontinuities; no intervention used before its known date; shrinking eligibility/capacity is not negative cumulative adoption. Known eligibility expansion with constant adoption propensity differs from a propensity increase; if aggregate data cannot distinguish them, expose non-identification or bounds. Existing users remain in their defined state unless a separate transition changes them.
Command: `cargo test --locked -p innovate-core --test regime_boundaries`.
- [ ] Red/green known/no-change and observational-equivalence cases; compare smooth alternatives, review and commit.

### A07 — Networks, complements and path dependence
Dependencies: A04, E01. Create `crates/innovate-core/src/models/network.rs`, `crates/innovate-kairos/tests/network_limits.rs`.
Interface: `InnovationBehaviour` consumes explicit topology/exposure and emits adoption/cessation actions into Kairos; model parameters remain Innovate-owned.
Test: no-edge/no-complement limits; node-permutation invariance; topology changes and exposure history; hysteresis sweep compared with independent small-state solution; one clock/RNG authority.
Command: `cargo test --locked -p innovate-kairos --test network_limits`.
- [ ] Red/green limited topology slice, then expand with Microfinance/temporal-network evaluation; do not equate an exposure coefficient with a causal spillover.

### A08 — Structural demand and innovation-specific strategic models
Dependencies: A04, G02. Create `crates/innovate-core/src/models/demand.rs`, `crates/innovate-games/src/market_game.rs`, `crates/innovate-games/tests/demand_game_reference.rs`.
Interface: `choice_shares(&ChoiceSet,&PreferenceParameters)->Result<Vec<f64>,InnovationError>` and `build_launch_game(&InnovationMarket)->Result<GameSpec,InnovationError>`.
Test: outside option and shares sum correctly; independent logit elasticities; dominated launch strategies removed only under justified assumptions; tiny timing-game equilibrium agrees with analytic solution. Prices are endogenous absent an identification design. Equilibrium predictions and fitted noisy-choice/inertia/learning models have distinct labels and validation.
Command: `cargo test --locked -p innovate-games --test demand_game_reference`.
- [ ] Start with logit/static/two-period cases; register BLP, real options, bounded-rationality and dynamic-oligopoly extensions separately with code/licence and identification audits, not one opaque solver task.

## Hype — healthcare_hype_validation_20261107

### H01 — Correct composite shape and derivatives
Dependencies: R02. Create `crates/innovate-core/src/models/hype.rs`, `crates/innovate-core/tests/hype_shape.rs`.
Interface: `hype_mean(t:f64,p:&HypeParameters)->Result<f64,InnovationError>` and `hype_derivative(...)`; parameters include baseline, free maturity amplitude, growth, timing and pulse widths/amplitudes.
Test: no-pulse limit is maturity; derivative matches independent finite difference; zero/negative widths and NaN rejected; combined extrema solve derivative of the sum; no hard clipping of latent expectation. Distinct ordered-cycle and unrestricted-wave configurations.
Command: `cargo test --locked -p innovate-core --test hype_shape`.
- [ ] Red/green tests before changing legacy visual output; verify smoothness/limits, review and commit.

### H02 — Joint composite likelihood and measurement identification
Dependencies: H01, A02. Create `crates/innovate-core/src/inference/hype_fit.rs`, `crates/innovate-core/tests/hype_fit.rs`.
Interface: `fit_hype(&HypeFitRequest)->FitOutcome`, returning component parameters, total predictions, measurement-anchor metadata and separate identification assessments; training-only time normalisation is stored. Multi-indicator extension uses `y_k=a_k+lambda_k*h+error_k` with explicit indicator errors and scale/sign/location constraints.
Test: identifiable noiseless recovery from several starts; indistinguishable decompositions flagged while total prediction can remain useful. Optional variable projection only for identity-Gaussian case. Before H04 joint fitting, demonstrate h*=2h/beta*=beta/2 leaves adoption unchanged; reject an unanchored joint parameterisation or report its identified set. Fix one declared loading/scale and latent location/sign convention, then independently verify the corresponding transformation. Do not use holdout normalisation or prior tightness as evidence of data identification.
Command: `cargo test --locked -p innovate-core --test hype_fit`.
- [ ] Red/green multistart and identification tests, compare an independent likelihood oracle, review and commit; stop if no defensible anchor exists.

### H03 — Native Bayesian composite and partial pooling
Dependencies: H02, A03, I04. Create `crates/innovate-core/src/inference/hype_posterior.rs`, `crates/innovate-core/tests/hype_posterior.rs`.
Interface: `sample_hype_posterior(&HypePosteriorRequest)->FitOutcome` using qualified native sampler/gradients and optional group effects; export anchor and measurement uncertainty.
Test: transforms/Jacobians correct; weak/no-cycle prior-predictive cases allowed; non-convergence rejects posterior promotion; data-dependent SBC and repeated coverage retain failures. Report prior sensitivity for weak identification. Preserve model-family uncertainty separately from a precise posterior within a single chosen curve; no automatic unsupported averaging weights.
Command: `cargo test --locked -p innovate-core --test hype_posterior`; execute I04 protocol with this model.
- [ ] Red/green contracts, native sampler qualification and calibration before Bayesian claims; review priors/identifiability and commit.

### H04 — Separate hype effects on adoption and use
Dependencies: H02, A01, A05. Create `crates/innovate-core/src/models/hype_effects.rs`, `crates/innovate-core/tests/hype_effects.rs`.
Interface: `innovation_rates(t:f64,state:&InnovationState,hype:&HypeSignal,p:&HypeEffectParameters)->Result<InnovationRates,InnovationError>` with separate log-p/log-q coefficients, lag configuration and optional cessation hazard; HypeSignal carries its measurement scale/anchor.
Test: zero coefficients recover baseline; negative effect inhibits uptake; cumulative first adoption cannot fall; cessation changes active stock; separate effects do not silently become a shared time multiplier. Scaling signal and inversely scaling effects preserves rates but cannot be represented as independently identified parameters without H02's anchor.
Command: `cargo test --locked -p innovate-core --test hype_effects`.
- [ ] Red/green mechanisms, explicit assumptions/recoverability, review and commit. Hype is expectations/attention impact, not a claim-exaggeration prerequisite.

### H05 — Delay semantics and endogenous feedback
Dependencies: H04, A06. Only a separately declared agent/dynamic execution case additionally requires E01. Create `crates/innovate-core/src/models/expectation_feedback.rs`, `crates/innovate-core/tests/expectation_feedback.rs`; audit legacy `src/innovate/hype/delayed_hype_bass.py`.
Interface: separate `ExternalLagSignal` from `DelayedStateModel`; feedback uses explicit hype/adoption equations and measurement models.
Test: zero lag reduces to baseline; shifted forcing agrees with a non-DDE reference; genuine state delay requires history; no-effect/reverse/shared-shock examples expose identification limits. Core ODE fitting must work without Kairos.
Command: `cargo test --locked -p innovate-core --test expectation_feedback`.
- [ ] Audit equation first, red/green one mechanism at a time; retain experimental status without adequate data, review and commit.

### H06 — General hype impact benchmark
Dependencies: H02, H04, V02. M05 applies only to the optional medicines cohort. Create `validation/hype/protocol.json`, `validation/hype/run.py`, `tests/validation/test_hype_protocol.py`.
Interface: frozen general-purpose manifest includes synthetic and qualified non-medical cases, no-cycle/single/repeated-wave families, flexible spline/state-space and no-hype adoption comparators; native outputs are orchestrated only.
Test: holdout technology/time is never used for selection; no-peak data is not forced into a cycle; curve fit, effects and incremental prediction are separate outcomes. Unavailable medicines data does not block the general suite. Record measurement/structural uncertainty, null results and failures; unsupported causal effects remain unsupported.
Command: `uv run pytest tests/validation/test_hype_protocol.py -q`, then versioned native benchmark jobs.
- [ ] Freeze/test protocol, run comparisons, review calibration/incremental scores and commit; no claim-exaggeration annotation prerequisite.

## Medicines and datasets — empirical_validation_publication_20261107

### M01 — Exact existing source manifest
Dependencies: none. Create `validation/data/pbs-sources.json`, `tests/validation/test_pbs_manifest.py`; consume the six paths/hashes in `research/medicines-utilisation-20261107.md`.
Interface: manifest identifies HF revision, digest, upstream receipt, rights/sensitivity scope and separate byte/semantic status. No new harvester.
Test: six unique exact objects; changed hash/revision rejected; later rights assessment not overwritten by historical unknown labels; absent modelling qualification remains absent.
Command: `uv run pytest tests/validation/test_pbs_manifest.py -q` with local receipt fixtures, not live network in fast CI.
- [ ] Red/green manifest, permitted acquisition in separately logged job, review and commit.

### M02 — Source-faithful Polars projection
Dependencies: M01, R03. Create `crates/innovate-data/src/atlas_import.rs`, `crates/innovate-data/tests/atlas_import.rs`; parsing/qualification changes go upstream to Atlas.
Interface: `read_atlas_projection(path:&Path,manifest:&SourceManifest,binding:&ColumnBinding)->Result<ObservationPanel,DataError>` accepts qualified Arrow/Parquet or native CSV; qualified Rust workbook conversion stays upstream. Retain lazy projection until required materialisation.
Test: fields/dtypes/units preserved; duplicates/invalid date basis rejected; absent fields remain null; no pandas Excel path or invented worksheet schema.
Command: `cargo test --locked -p innovate-data --test atlas_import`.
- [ ] Red/green bounded fixture, upstream canary/semantics evidence, review and commit; filenames alone never establish coverage.

### M03 — Bitemporal mapping and observation-process qualification
Dependencies: M02. Create `crates/innovate-data/src/utilisation_panel.rs`, `crates/innovate-data/tests/utilisation_panel.rs`.
Interface: `qualify_utilisation(&ObservationPanel,&MappingHistory)->Result<QualifiedPanel,QualificationError>` preserves effective dates, known-at dates, source vintage and recording/supply definitions.
Test: mapping effective Jan1 but first known Mar1 is excluded from a Feb1 forecast, even if later backfilled; as-of joins enforce both clocks. Preserve unmapped codes; avoid annual/detail double counting; separate scripts/stock/spend; flag pack/code/coverage/reporting changes and missing months. A stable treatment-exposure fixture with halved dispensing frequency is an observation change, not automatic abandonment. Never infer unique patients.
Command: `cargo test --locked -p innovate-data --test utilisation_panel`.
- [ ] Red/green tests, actual coverage/grain report, review and commit before modelling.

### M04 — Generic utilisation observation and eligibility model
Dependencies: M03, A05, A06. Create `crates/innovate-core/tests/utilisation_identifiability.rs` and `examples/medicines/utilisation-model.json`.
Interface: application binds active use, supply intensity, eligibility M(t), coverage and source measures; generic methods remain A05/A06-owned. Explicitly map payer/institution/prescriber/patient/dispensing decisions only where measured or modelled; no requirement that each be an agent.
Test: known starts/continuation/switches recover under sufficient measurements; different latent histories with equal scripts expose non-identification. Constant100 active users with supply interval1 versus2 yields expected transactions100 versus50 without a stock decline. Eligibility expansion and access/propensity changes are separately tested; indistinguishable cases return bounds/scenarios. No automatic utility/patient conversion.
Command: `cargo test --locked -p innovate-core --test utilisation_identifiability`.
- [ ] Red/green identification/bounds tests and a non-medical repeat-use example, review and commit.

### M05 — Frozen medicines forecast cohort
Dependencies: M03, A01, V02. Create `validation/medicines/protocol.json`, `validation/medicines/run.py`, `tests/validation/test_medicines_protocol.py`.
Interface: freeze actual coverage, cohorts, mappings, origins, horizons, source vintages and separate price/quantity outcomes; preserve effective/known-at dates and eligibility/supply regimes.
Test: training-only saturation/scaling; no holdout tuning or later mapping revisions; overlap handled; sparse/new/zero series retain explicit eligibility. Insufficient history fails the proposed 24-month-training/12-month-holdout design rather than padding it. Include supply-interval/coding/coverage shocks and time-varying eligibility sensitivity.
Command: `uv run pytest tests/validation/test_medicines_protocol.py -q`.
- [ ] Freeze after source pilot, compare named baselines/candidates, review paired uncertainty/failures and commit results.

## Game-runtime boundary — advanced_diffusion_frontier_20261107

### G01 — Inventory and consolidate generic game ownership
Dependencies: none. Provider: `edithatogo/paignia`, with seed transfer from `new-drug-reimbursement-game` and T17.01-T17.17 reconciliation pending. Coordinate Kairos track110 S01-S04. Inspect `crates/paignia-core`, `crates/paignia-solve` (historically `uogto-game-*`) and Kairos `crates/kairo-ecs-game-{theory,ontology}`; extend provider `extraction-manifest.json`, `docs/architecture/paignia-transfer-map.md` and existing guards rather than duplicate inventories.
Interface: one authoritative GameSpec/Policy/Solution contract; UOGTO supplies semantic validation; Kairos owns execution; no second scheduler.
Test: static graph contains no application or mandatory simulator/ontology runtime; every advertised method has one owner and explicit accepted/unsupported/migration state; no obligation disappears.
Command: provider `cargo test --workspace --locked` plus its ownership-graph test.
- [ ] Inventory callers before moving code; retain compatible adapters/provenance, review and commit; repository creation is not extraction evidence.

### G02 — Independently consumable solver slice and honest solution contract
Dependencies: G01, R08. Paignia files: `crates/paignia-core`, `crates/paignia-solve`; future `crates/paignia-solve/tests/solver_oracles.rs` and installed-consumer fixtures. Use the exact transferred artifact, not an assumed registry release.
Interface: `solve_finite_game(&GameSpec,&SolveConfig)->Result<Solution,SolveError>` reports game class, information structure, solution concept, algorithm/version, status, tolerance, residual/exploitability, selection rule and completeness claim. Separate one-found, exhaustive-within-declared-scope, approximate and behavioural results.
Test: named independent cases per claimed algorithm. Backward induction uses supported trees; pure-Nash enumeration covers dominance, coordination and matching pennies with zero pure equilibria. Mixed-Nash, CE constraints and CFR/exploitability gate only those claimed methods; unfinished methods retain T17.04-.07/S01-S04 obligations and explicit unsupported status. Test rectangular/invalid games and incomplete catalogues. Noisy-choice/inertia/adaptive-learning predictions require their own model labels and empirical tests, not a Nash label.
Command: provider `cargo test --locked -p paignia-solve --test solver_oracles`.
- [ ] Use isolated Gambit/OpenSpiel references; qualify reimbursement and Innovate installed consumers; one algorithm per reviewed slice.

For Innovate's static consumer, create `crates/innovate-games/Cargo.toml`, `src/lib.rs`, `tests/static_provider.rs` under that crate and root workspace/lock entries in a separate leaf of at most five files. The qualified core/solver does not require Kairos. An actual innovation-game test checks algorithm identity, unsupported cases and result semantics; run `cargo test --locked -p innovate-games --test static_provider`. This establishes A08's adapter; G03 extends it only for optional execution.

### G03 — Optional dynamic adapter and lawful policy observation
Dependencies: G02, E01. Paignia owns optional `crates/paignia-kairos/{Cargo.toml,src/lib.rs,tests/replay.rs}`; Innovate adds an optional execution feature to the existing G02 adapter. Reuse T17.08/.14/.15/.17 and current Kairos observation/RNG contracts.
Interface: game actions/observations use Kairos public APIs; policy inputs contain permitted observations/legal actions/identity only, never full world references. Static algorithms accept explicit RNG configuration independently of Kairos.
Test: same lawful observation in two hidden states yields identical policy input/action under matched policy randomness. Opponent-private data, leaked action masks and future events must fail. Solver exploration must not alter paired exogenous simulation streams. One clock/state/RNG, deterministic replay and disabled-feature closure tests remain required; unsupported information sets fail explicitly.
Command: provider `cargo test --locked -p paignia-kairos --test replay`, mapped T17 observation/RNG targets and Innovate installed-adapter smoke.
- [ ] Red/green adapter at public boundaries, immutable provider pin and trace evidence, review and commit. Offline game inspection does not grant live policies that access.

### G04 — Static and dynamic strategic innovation research cases
Dependencies: static cases require A08, G02, V02. Only cases requiring dynamic execution additionally require G03. Create `validation/competition/protocol.json`, `validation/competition/run.py`, `tests/validation/test_competition_protocol.py`.
Interface: launch timing, price/quality, switching, cannibalisation and optional investment/learning cases declare game class, information structure, equilibrium/behavioural interpretation and execution requirement.
Test: replicate a published/analytic special case before extension; justify payoff transformations; preserve multiplicity/selection sensitivity. Static cases run with Kairos absent; dynamic cases fail readiness without qualified G03. Do not treat solver restart frequencies as behavioural probabilities.
Command: `uv run pytest tests/validation/test_competition_protocol.py -q` then native jobs.
- [ ] Audit literature and existing code/licence per case, report static/dynamic results separately and retain unfinished obligations. Pending dynamic work does not block a scoped static result, I10 runtime-only qualification or close the whole packet.

## Ecosystem integration — ecosystem_decision_integration_20261107

### E01 — Real Kairos execution
Dependencies: R02, R08. Create `crates/innovate-kairos/{Cargo.toml,src/lib.rs,tests/native_execution.rs}`; retire `src/innovate/abm/kairos_adapter.py` reference fallback from supported dispatch.
Interface: `run_innovation(&SimulationRequest,&KairosRuntime)->Result<SimulationResult,InnovationError>` uses qualified native event/state/RNG APIs; Innovate supplies behaviours.
Test: actual native identity, no Python reference call, missing provider fails, deterministic replay/resource conservation, no second clock. Keep solver/policy/exogenous random-purpose identities distinct under declared coupling.
Command: `cargo test --locked -p innovate-kairos --test native_execution`.
- [ ] Red/green runtime and installed consumer, qualified pin, review and commit.

### E02 — Uncertainty-preserving Voiage handoff and noise control
Dependencies: R04, A01. Create `specs/innovation-decision-binding-v1.json`, `tests/integration/test_voiage_handoff.py`; the numerical/input adapter belongs in Voiage's existing optional integration work.
Interface: DecisionBinding maps joint outcomes to strategy-labelled utilities/net benefits, perspective, population, horizon and discount basis. Preserve parameter/model/equilibrium/measurement/simulation-replicate identities and source assessments. Voiage owns conditional expectation and VOI arithmetic.
Test: correlations survive; swapped labels/units/invalid fits rejected; equal-probability parameter rows A=[4,0], B=[1,3] give conventional EVPI1.5. With ONE parameter state and two equally weighted simulation replicates A=[-1,1], B=[1,-1], conditional means are0 and parameter EVPI0; treating replicates as parameter states gives1 and must fail. Include heteroscedastic and unequal-mean cases, Monte Carlo error/bias diagnostics and coupling metadata. Raw prescriptions without valuation cannot become utility.
Command: `uv run pytest tests/integration/test_voiage_handoff.py -q` against installed artifacts; use independent finite arithmetic, not provider-generated expected values.
- [ ] Red/green contract and actual native consumer, preserve inner/outer budgets and MC error, review and commit. No mandatory reverse dependency.

### E03 — Information estimands and decision feedback without cycles
Dependencies: E02. Strategic-response cases additionally require the relevant qualified Paignia method, not every optional game capability. Create `specs/study-design-v1.json`, `tests/integration/test_study_design_feedback.py` and an application orchestration example.
Interface: distinguish conventional EVPI/EVSI, net benefit of research, strategic information value and social information value. StudyDesign/PolicyScenario records recipient, timing, public/private disclosure, commitments, observation model, solution concept, selection rule and explicit welfare mapping. Reuse Voiage's signed-social information contract and preserve/version legacy semantics.
Test: acyclic dependencies; study observations follow the specified model; reproducible declared coupling. Fixed-policy case reduces to ordinary VOI; supported recomputed-response case retains a negative strategic value without clipping or calling it conventional EVPI. Unsupported/nonconverged results are not economic zero. Generic variance reduction and averaging fully observed hidden-state games fail as substitutes for an information-consistent study.
Command: `uv run pytest tests/integration/test_study_design_feedback.py -q` plus provider SI01-SI04/SI-V01-SI-V08 targets once available.
- [ ] Red/green finite independent examples, review estimand/information structure, commit; mechanism/information-design research remains a distinct experiment.

### E04 — Promote VOP proof-of-concept capabilities to Voiage
Dependencies: no Innovate-adapter prerequisite for independent method inventory/promotion; provider-specific prerequisites govern each slice. E02 is required only when that slice consumes Innovate outputs. Provider: Voiage. Inspect `vop_poc_nz` CEA/DCEA/VOI/perspective/Sobol helpers and `voiage/rust/crates/{voiage-domain,voiage-numerics}`; create extraction ledger and `rust/crates/voiage-numerics/tests/vop_reference.rs`.
Interface: generic perspective/equity/decision/information methods are Voiage-owned; NZ scenarios remain local; simulation is Kairos-owned where needed and adoption Innovate-owned.
Test: independent small perspective/regret/equity fixtures; preserve or justify changes to application outputs; POC calls provider, not copied kernels. Never make a flawed POC the numerical oracle.
Command in Voiage: `cargo test --manifest-path rust/Cargo.toml --locked -p voiage-numerics --test vop_reference`; installed POC consumer.
- [ ] Inventory and verify each method, retire duplicate framework claims, preserve rollback and migration evidence, review and commit.

### E05 — Application promotion and surrogate gate
Dependencies: inventory is independent; each implementation needs only its actual providers. A simulation surrogate needs E01; decision-preservation evaluation needs E02; a medicines-specific example needs M04. Create `conductor/qualification/application-promotion.json`, `tests/integration/test_application_contracts.py`; coordinate tobac/CareOps/MARS without copying domain code.
Interface: reusable method ledger maps owner, domain residue, independent oracle and installed consumer. MARS approximates explicitly identified expensive functions only.
Test: distinct applications consume the same provider method; no library imports applications; surrogate held-out error and decision ranking meet frozen tolerances; out-of-domain use fails/abstains. Unrelated application/data readiness cannot block method inventory or a different qualified consumer.
Command: `uv run pytest tests/integration/test_application_contracts.py -q` plus provider-native tests.
- [ ] Red/green handoffs, retain negative surrogate decisions and costs, review and commit.

## Validation and publication — empirical_validation_publication_20261107

### V01 — Independent model-oracle and invariant registry
Dependencies: R02. Create `validation/oracles/manifest.json`, `tests/validation/test_oracle_manifest.py` and classical/substitution/competition fixtures.
Interface: record independent derivation/package/version, licence, inputs and prespecified tolerances. A fixture generated solely by the tested kernel is not independent.
Test: same implementation hash cannot claim independence; stale inputs/unknown tolerance fail; known wrong outputs caught. Add product relabelling, consistent time/rate conversion, justified positive affine utility transforms and partition/recombination invariants where mathematically applicable. Do not demand an invariant that changes the estimand.
Command: `uv run pytest tests/validation/test_oracle_manifest.py -q`.
- [ ] Red/green registry, independent reference acquisition in isolated environments, review and commit.

### V02 — Leakage-safe benchmark runner
Dependencies: R04, V01. Modify `src/innovate/benchmarks/runner.py`, `src/innovate/benchmarks/corpus.py`; create `tests/validation/test_temporal_benchmark.py`.
Interface: BenchmarkProtocol fixes source vintages/effective/known-at dates, origins, horizons, tuning budget, native candidate and independent references; runner orchestrates only.
Test: future scaling/saturation/revisions and duplicate holdouts rejected; failures remain in denominators; seeds/coupling recorded; output includes proper scores, calibration, time/memory and MC precision. More numerical effort must not silently redefine the estimand; report error reduction rather than assert monotonicity for every random run.
Command: `uv run pytest tests/validation/test_temporal_benchmark.py -q`.
- [ ] Red/green adversarial leakage/noise checks, review and commit before final holdout examination.

### V03 — Cross-domain, early-stage and model-discriminating validation
Dependencies: V02, A01. Create `validation/cross-domain/protocol.json`, `tests/validation/test_cross_domain_protocol.py`.
Interface: qualified CHAT/CER, IBM/DIMORA and policy/network cohorts preserve actual observation semantics, rights, revision and coverage. Each method is evaluated only within qualified scope.
Test: leave-technology/region-out and pre-inflection splits; usage/publication counts not relabelled new adopters; weak market-potential identification reported. Add known-truth competing mechanisms with identical aggregate trajectories and a prespecified extra measurement that can discriminate them; compare simple baselines and retain failure to distinguish.
Command: `uv run pytest tests/validation/test_cross_domain_protocol.py -q`; frozen native/reference jobs.
- [ ] Freeze protocol, evaluate calibration/failures/negative findings, review and commit. Any research-value claim routes through E02/E03 rather than predictive scores alone.

### V04 — Independent native performance and optional integration experiments
Dependencies: native workloads require R07, V02. Only declared Kairos integration workloads additionally require E01. Create `validation/performance/protocol.json`, `crates/innovate-core/benches/workloads.rs`, `tests/validation/test_performance_receipts.py`.
Interface: equivalent numerical work with precision, cold/warm time, compilation, transfer, memory, hardware and accuracy; optional ensembles/parallel/GPU/native-simulation cases separately identified.
Test: fake device/metadata-only execution rejected; different work/precision cannot be a speedup; retain slowdowns/crossovers. Native benchmark readiness remains true when an unclaimed optional simulator is unavailable; claimed integration work fails without its execution evidence.
Command: `cargo bench --locked -p innovate-core --bench workloads`; `uv run pytest tests/validation/test_performance_receipts.py -q`.
- [ ] Sequential baseline before optimisation, measured bottlenecks only, review and commit separate receipts per scope.

### V05 — Reproduction and publication packages
Dependencies: V03, Q03 for the scoped software/general forecasting bundle. M05 is additional only for a medicines paper; each other manuscript requires its own claimed-method receipts. Modify `paper.qmd`, `references.bib`, `CITATION.cff`; create immutable paper manifests and `tests/validation/test_reproduction_bundle.py`.
Interface: clean-user reproduction from rights-permitted pinned inputs and installed native artifacts.
Test: changed/missing inputs fail; tables match computation; no restricted redistribution; AI assistance/versions disclosed; unrelated studies are not duplicate publications. An unfinished optional application is not a prerequisite for an honest scoped software paper.
Command: `uv run pytest tests/validation/test_reproduction_bundle.py -q` and recorded clean reproduction.
- [ ] Separate JOSS software, forecasting, innovation/hype, decision and data outputs; verify venue rules; review and commit. External submission requires separate authorisation.

## Hardening and release — empirical_validation_publication_20261107

### Q01 — Malformed-input, property and FFI safety gates
Dependencies: R04, R09. Create `fuzz/`, `crates/innovate-core/tests/properties.rs`, `bindings/ffi/tests/adversarial.rs` and bounded CI targets.
Interface: malformed native inputs produce typed errors, not FFI panics; deterministic ownership.
Test: overflow, oversized payload, invalid UTF-8/schema, null/duplicate keys, release-after-error, cancellation/interrupted files, seeded corpus replay and conservation/invariants.
Command: `cargo test --workspace --locked`; named bounded fuzz runs record corpus/hash/stop conditions.
- [ ] Red/green faults, sanitizers where supported, review and commit; one run is not universal memory-safety proof.

### Q02 — Native and no-legacy CI matrix
Dependencies: R06, R07, R09. Modify `.github/workflows/ci.yml`, binding/package workflows; create `tests/installed/test_release_matrix.py`.
Interface: declared platform/architecture/native artifacts with actual installation and method execution.
Test: Windows x64/Linux x64/macOS arm64 where advertised; missing artifact and pandas/xarray reintroduction fail; an unsupported/skipped lane cannot establish support. Test minimal and declared feature combinations in independent consumer workspaces.
Command: `uv run pytest tests/installed/test_release_matrix.py -q`; hosted results bound to exact head.
- [ ] Red/green matrix, retain stronger gates, review security and commit.

### Q03 — Supply chain and reproducible artifacts
Dependencies: R08, Q02. Modify release workflows/packaging; create `conductor/qualification/release-artifacts.json`, `tests/installed/test_artifact_provenance.py`.
Interface: hashes, lock/toolchain/source, SBOM/licence/security and actual publishability per package. Candidate testing is not release-lock acceptance.
Test: tampered hash, unapproved licence, missing native library or unpublishable closure blocks candidate; clean wheel/crate consumers reproduce declared examples. Preserve feature-specific dependency closure and immutable provenance.
Command: `uv run pytest tests/installed/test_artifact_provenance.py -q`; audit/deny/package tools with recorded exact versions.
- [ ] Red/green corruption/packaging checks, inspect built artifacts, review and commit; no publication in this task.

### Q04 — Scientific and documentation release truth
Dependencies: R10, I04, V03, Q01. Create `tests/validation/test_release_claims.py`; generate capability/support docs from evidence.
Interface: enumerate stable/experimental/unavailable combinations and outstanding obligations; separate execution/convergence/identification/prediction/causal assessments.
Test: stale/wrong-source receipt revokes acceptance; fake posterior/prior-only/wrong-method mutations caught; unqualified hype/network/game scope not stable; installed examples run. A green schema check cannot stand in for numerical/empirical results.
Command: `uv run pytest tests/validation/test_release_claims.py -q`; coverage/mutation checks per execution contract.
- [ ] Red/green false-claim cases, review migration warnings and commit; never delete unresolved requirements to obtain green status.

### Q05 — Independent beta, RC and stable acceptance
Dependencies: Q02, Q03, Q04, V04 native scope. E02 is required only for an advertised Voiage integration; E01/G03 and their receipts are required only for advertised dynamic capabilities. Modify `conductor/qualification/release-decision.json`, checklist and changelog.
Interface: separate native-core, data-bindings and optional-integration decisions identify exact candidate, supported scope, mandatory receipts, blockers and publication authorisation. Preserve all unfinished obligations outside the claimed scope.
Test: fail/skip/unrun/expired mandatory gate blocks its scoped promotion; historical green cannot qualify new source; full-native claims require every advertised stable combination. A core release without Voiage/Kairos is eligible only if all core/data/binding claims pass and optional capabilities are clearly unavailable/experimental; an advertised integration without its receipt must fail.
Command: existing release-readiness workflow plus actual installed matrix on the exact release commit, with scope-specific negative controls.
- [ ] Beta use/reproduction, RC hardening and stable decisions in order; retain experimental scope and existing obligations; review and commit. Publication remains separately authorised.
