# ABM and forcing continuation review

Same isolated writing lane, based on 3cc10543. Kairos ABMContext owns entity
creation, component storage and event dispatch. A dedicated Kairos identity RNG
selects susceptible agents; the existing timing stream remains unchanged for
homogeneous runs without forcing. Tests verify unique/conserved identities,
monotonic irreversible adoption, replay and equality with DES counts.

The simulation-only forcing contract explicitly uses right-continuous,
piecewise-constant additive covariates and cumulative event rate deltas. Unknown
fields, invalid shapes/order, negative effective rates and colliding tick
boundaries fail. Zero pressure waits for the next rate change. Boundary-first
ordering and exponential memoryless resampling are documented. Independent
binomial ensemble means cover constant innovation and integrated forced hazards.
This is not full historical linear-covariate, market-potential, fitting or
model-specific event-split parity; those features remain required in the ledger.

Local self-review resolved the upstream RNG API mismatch by using Kairos'
reexported Uniform distribution, without introducing a second RNG implementation.
Clippy suggestions were applied without suppressions. No agent, network,
intervention, simulation-resume or cross-platform feature is inferred from a
passing homogeneous case. Agent records represent state at the last sample.

All current/planned obligations remain required. The registry and actual active
track directories both contain one active track: Scientific ecosystem alignment.
Archived tracks remain historical records of older implementations.

Validation: 45 Rust tests (41 core/C ABI in coverage, 4 Polars), 24 installed
Python tests, 135 Julia assertions and R native checks. Lint/types, Clippy,
163-page docs build, wheel/sdist metadata and audit pass within recorded scope.
The audit retains the existing unsuppressed bincode maintenance warning.
Overall core/C ABI line coverage is 80.18%; branch/mutation, clean-source package,
platform distribution and full scientific qualification remain open. This local
checkpoint does not complete the track or authorise main promotion/publication.
