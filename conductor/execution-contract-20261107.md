# Executable work contract for bounded coding agents

Read `product.md`, `frontier-programme-20261107.md`, the owning track spec and the named task in `innovation-taskbook-20261107.md`. The taskbook replaces vague earlier implementation steps where they conflict; historical receipts remain intact.

## One work packet

1. Select a task whose dependencies have accepted receipts. A pending upstream task is a blocker, not permission to invent a substitute.
2. Read only the specified interfaces plus their callers/tests. Create an isolated branch/worktree. Record base commit, task ID and allowed files. One integrator owns root manifests, pins, shared schemas and registries.
3. Write the named failing test and its specified assertions. Run the listed command and retain its real failing output. A missing test target is not a mathematical regression; create the target first where instructed.
4. Implement only the named interface and minimum required behaviour. No fake posterior, identity inference, synthetic production outputs or host-language fallback.
5. Rerun focused tests, then affected contract and installed-consumer tests. Save actual outputs, exact revisions/lock hashes, seed, environment and limitations. Never mark skipped, cancelled, timed-out or unrun checks as passes.
6. Review against the task's oracle and scope. A more capable independent agent may review when available; otherwise report self-review accurately. Scientific sign-off is not inferred from a generic code reviewer.
7. Commit one testable slice. Record test receipt and commit in the owning track. Integrator runs required CI before promotion. Do not silently change a tolerance, fixture or acceptance criterion to pass.

## Failure handling

Two unsuccessful targeted repair attempts trigger a blocker record with failing command, observed/expected result, minimal reproducer and upstream owner. Continue independent ready tasks, not dependent tasks. Stop for unsafe/destructive ambiguity, missing lawful data, credentials or a scientific contract requiring a new decision. Do not ask the user questions already answered by the plans.

## Shared interfaces

`innovate-core` exposes typed `ModelId`, `Parameters`, `TimeGrid`, `ObservationSeries`, `FitRequest`, `FitOutcome`, `PredictionRequest`, `PredictionBundle` and `InnovationError`. `FitOutcome` is `Valid(FitResult)` or `Invalid(FitFailure)`; posterior access requires a valid posterior status. Observation kind explicitly distinguishes new adoption, active stock, transactions, rate, share and attention/expectation observations.

`innovate-arrow` owns versioned Arrow schemas and `JointDrawBundle`: draw/chain/scenario/strategy/product/region/time IDs, values, units, population/denominator, source vintage, fit status and provenance. `innovate-data` accepts native Polars frames and uses explicit column bindings; no heuristic dataframe conversion. APIs below are target contracts, not claims that types exist yet. R01/R02/R03 create them before dependants run.

## Test policy

Deterministic reference tests begin with `abs(error) <= 1e-8 + 1e-6*abs(reference)` for well-scaled examples; each algorithm records a justified tighter/different tolerance before final evaluation. Conservation and invalid-input requirements are exact where mathematically appropriate. Statistical tests use declared Monte Carlo error and repeated seeds; a single favourable random run is not acceptance. Prediction and causal inference have distinct receipts.

New production code targets at least 90% line coverage and meaningful branch/negative coverage; retain any stricter repository rule. Mutation cases must include wrong-method labels, unsupported native fallback, broken draw dependence, invalid likelihood and incorrect units. Coverage cannot substitute for an independent mathematical oracle.

## Cross-repository tasks

E04/G01/G02 are handoffs with named owners. Their provider implementation belongs in Voiage or the game-runtime provider, not copied into Innovate. Record provider PR, accepted immutable revision and installed-consumer test before closing the Innovate handoff. A tracking issue alone is not implementation acceptance.
