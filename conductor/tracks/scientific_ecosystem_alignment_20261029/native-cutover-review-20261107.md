# Native cutover review — 7 November 2026

The isolated 0.7 development line executes scientific work in Rust and uses
Polars/Arrow IPC at the table boundary. Python is a PyO3 transport; Julia and R
use the native C ABI. The other language packages and Python scientific runtime
were removed. This is a substantial breaking checkpoint, not full migration or
release completion. The user's explicit permission allows interim unavailability
while preserving every current and planned scientific requirement.

## Evidence and scientific corrections

The final source/artifact hashes and retained logs are in native-cutover-20261107.json
and evidence/native-cutover-20261107/. Checks used repo-local Python 3.15.2 on
macOS arm64, Rust 1.99.1, Julia 1.13.3 and installed R 4.3.0. The clean sdist was
unpacked by Maturin and built into the installed native wheel. Only Polars is a
base Python runtime dependency; pandas/xarray and the old scientific stack are
absent from that environment. Optional PyArrow stays optional.

Seventeen Rust tests, ten installed Python tests from outside the checkout,
114 Julia assertions and installed R analytic/error/repeat-call checks passed.
The five named models have independent closed-form prediction oracles; selected
noiseless fits recover parameters. Bass also recovers parameters with irregular
times and rescaled units. The Voiage 3.1.0 installed consumer matches analytic
EVPI after a Polars/Arrow IPC handoff. This is synthetic CPU evidence only.

Observed regressions preceded fixes: Python fallback/unported discovery claims,
ignored Gompertz `b`, fabricated significance p-values, invalid domains/time,
near-saturation asymptote search bounds, Bass absolute-time forecast batches,
Bass fitting failure on an analytic curve, inconsistent information-criterion
parameter counts and inconsistent direct-parameter payloads across models.
Rust now resolves direct parameters versus fitted state uniformly and rejects
ambiguous requests. AIC/BIC record the iid-Gaussian convention, include residual
variance in the parameter count, omit the common Gaussian likelihood constant,
and return null where undefined. Unknown significance tests are null, not p=1.
None of these checks establishes general convergence, calibration or identifiability.

## Dependency and tooling review

The exact upstream Polars source at 2d74c435e83b86dc2cf6b71ec6c7b4bd67a4042d
and its hashbrown patch remove the vulnerable quick-xml graph resolved by Rust
Polars 0.55.2. Cargo audit reports zero vulnerabilities, retaining the unsuppressed
bincode unmaintained warning RUSTSEC-2025-0141. pip-audit reports no known
vulnerabilities among audited dependencies; unpublished local Innovate is skipped.
Latest registered candidates and specific qualification exceptions are retained
in dependency-frontier-20261106.json. TypeScript 6.0.3 remains for Astro tooling
because current Astro/Volar reject 7; this is not a TypeScript library binding.
The pnpm release-age guard is retained. Matching stable emnapi core/runtime fixes
the failed documentation checker without suppressing the failure.

Rust Clippy, native Python lint/types/security, selected ordinary pre-commit
hooks, actionlint, Twine, wheel-content checks and the 163-page docs build pass.
The native extension has a type stub; the source-only type checker retains one
expected missing-native-source warning. Astro check has zero errors/warnings.
No shared Git hook was installed or disabled. The superseded type-checker config
was removed so the check actually covers the new binding source. Conda is now a
platform-specific native recipe; it is not build-qualified. Publication workflows
and local release helpers enforce the feature ledger before proceeding.

The all-workflow Zizmor read found four cache findings in changed release lanes;
explicitly disabling those release caches resolved them. Two pre-existing
trigger findings remain in unchanged ci-gate-monitor.yml and tagbot.yml and
require their owners' review. This checkpoint does not assert a general security
clearance and did not activate or run those workflows.

## Remaining acceptance and source preservation

All 16 model families and 22 feature areas remain unqualified for full migration;
five families expose bounded development operations. Native multi-generation
substitution, covariates/events, other model families, inference/uncertainty,
Kairos dispatch, preprocessing, scenarios, reporting, plugins and HPC remain
required. Native IPC is currently copied, and Python LazyFrames collect at the
execution boundary. Julia/R table IPC and bundled distributions remain open.
Linux, Windows, containers, Conda, hosted CI, mutation testing and broader
installed-platform acceptance have not been established here. No Docker/Podman
CLI was available for container qualification.

Measured core/C ABI line coverage is 71.50% (2444/3418), below the historical
80% release threshold. This excludes the tabular crate, language coverage and
unported features, with no branch instrumentation. The release guard intentionally
fails for all 38 obligations, including under Python optimization. Do not use the
smaller native suite to claim the retired Python tests now pass.

466 retired tracked paths are individually recorded with original SHA-256 and
source commit in retired-paths-final-20261107.json; replacement paths are not
counted as deletions. The source/test feature inventory retains requirements and
public definitions. The three earlier archive reconciliations and their remaining
human verification conditions are preserved. Old docs and planning records are
labelled historical, not promoted as current API promises.

## Alignment and next work

Use Voiage's separated Rust domain/numerics/serialization/binding boundaries and
installed Polars/Arrow consumer evidence. Keep VOI/economic calculations there.
Use Kairos for scheduling, DES/ABM state, deterministic streams and replay; its
latest inspected source is not yet a qualified Innovate runtime dependency.
Innovate owns diffusion semantics, fit quality and forecasting evidence.

Prioritise native multi-generation substitution, explicit optimizer convergence
and identifiability; time-held-out forecast scoring and interval calibration;
incident versus cumulative adoption semantics; paired seeded scenario comparisons;
missing/irregular-time policies; and provenance-bearing Arrow bundles. The source
inventory is the implementation input. The exact current Kairos adapter and
remaining native feature/acceptance work stay on the active track.

Review is a local source and evidence review by this writing lane, not independent
integrator acceptance. No main integration, push, registry release, external
message or native activation was performed. Other worktrees and environments
remain with their existing owners.
