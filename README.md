# Innovate

Innovate models innovation adoption, technology substitution and policy diffusion.
The 0.7 development line uses a **Rust scientific core**, **Polars** tables and
thin **Python, Julia and R** bindings. This is a breaking migration in progress.
All current and planned scientific features remain required; unported features
are temporarily unavailable and return explicit errors. There is no Python
numerical fallback. Rust is the core library, not a fourth binding.

## Current native scope

All sixteen model families now have bounded Rust implementations: Bass, Logistic,
Gompertz, Fisher–Pry, Norton–Bass, Lotka–Volterra, multi-product competition,
complementary goods, lock-in, network diffusion, policy diffusion, latent process,
regime switching, hierarchical, mixture and composite. Discovery describes the
currently executable operations and variants; the
[native migration guide](docs/astro-site/src/content/docs/native-migration.md)
records their scientific conventions, initial-state semantics and limits.

Current extensions include pointwise Logistic and mechanistic Bass/Gompertz
covariates/events with saved forcing history, native scalar
precision weighting, Gaussian mixture EM, coupled composite ODE levels/rates,
and bounded Kairos DES/ABM/network-resource simulation. Full family capability,
covariates/events across other models, uncertainty/inference, richer Kairos and
the remaining areas in [the migration ledger](migration-ledger.json) still require
implementation or qualification. All 38 obligations remain required and the
release gate intentionally fails. Local numerical convergence, rank and consumer
checks are development evidence, not full scientific or release acceptance.
The earlier Python suite remains in Git history as migration evidence; the native
suite does not claim equivalent feature coverage.

Polars DataFrames/LazyFrames are the Python table interface. Rust reads/writes
Arrow IPC and owns table validation and predictions. IPC currently copies buffers;
LazyFrames collect at the native execution boundary. R/Julia currently expose
the versioned JSON contract through a native C ABI. Their tabular IPC adapters
and distributable native-library bundles remain open.

## Native workflow discovery

Call `list_advanced_capabilities` through Python's `innovate.execute` or the
Julia and R `kernel_call` wrappers. The response lists the operation, stability,
and incremental or uncertainty support for each workflow.

Use `compose_regime_ensemble` to combine aligned forecasts by regime. Supply
`time`, a mapping of `predictions`, and optional `weights` and `observed` arrays.
Use `update_streaming_forecast` to append aligned cumulative observations with
`previous_time`, `previous_observed`, `new_time`, and `new_observed`.
The response contains the combined series and the growth since the previous state.

Call `render_residual_diagnostics_svg` with `time` and `residuals` to export
the residual, ACF, and PACF panels. The optional `config` accepts `title`,
`lags`, `acf_only`, `figsize` in inches, and a color for each panel.

## OECD observations

Use the agency and dataflow identifier and the ordered dimension key from
the OECD Data Explorer API query builder. Python returns a Polars DataFrame
with string period codes and nullable floating-point observations.

```python
from innovate import fetch_oecd_dataset

observations = fetch_oecd_dataset(
    "OECD.SDD.STES,DSD_STES@DF_CLI",
    ".M.LI...AA...H",
    start_date="2023-02",
    end_date="2023-03",
)
```

In Julia and R, call `fetch_oecd_dataset` through `kernel_call` with the same
payload fields. The native result contains CSV and the request metadata.
Omitted dates default to 1960 and 2020.

## Development

Use Rust/Cargo 1.99.1 and repo-local Python 3.15.2. Keep any earlier environment
until its active operations finish. Create the separate environment only for a
new checkout; reuse an existing owned environment and finish its active jobs
before changing dependencies:

```sh
uv venv .venv-native --python 3.15.2
UV_PROJECT_ENVIRONMENT=.venv-native uv sync --locked --python 3.15.2
.venv-native/bin/python -c 'import sys; print(sys.version, sys.prefix)'
cargo test --locked -p innovate-core -p innovate-tabular -p innovate-ffi -j 2
.venv-native/bin/python -m pytest tests/native
.venv-native/bin/python scripts/check_migration.py
```

```python
import polars as pl
from innovate import fit, predict

forecast = predict("logistic", pl.DataFrame({"time": [0.0, 1.0, 2.0]}),
                   {"L": 100.0, "k": 1.0, "x0": 1.0})
result = fit("logistic", pl.DataFrame({
    "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "observed": [4.7, 11.9, 26.9, 50.0, 73.1, 88.1],
}))
```

## Contracts and scientific changes

The new wire schema is **2.0**. Time columns must be finite and strictly increasing;
nulls, duplicate times and invalid parameter domains are rejected in Rust.
Predictions preserve row identity and never overwrite an existing prediction column.
Gompertz now uses the explicit standard curve `a * exp(-b * exp(-c * t))`;
the earlier implementation ignored `b`. Unimplemented significance tests return
null with `not_implemented`, never fabricated p-values. Synthetic parameter
recovery and independent ODE fixtures are tested. Local rank does not establish
global identifiability or calibrated uncertainty. See the
[native scientific conventions and examples](docs/astro-site/src/content/docs/native-migration.md).

The Rust operation `calibrate_prediction_intervals` accepts aligned `time`,
`observed` and `predicted` arrays, optional `confidence` (default 0.8), and an
optional numeric `holdout` mask. It returns symmetric empirical bounds and
descriptive coverage. Selected holdout observations can widen the interval,
so their reported coverage cannot serve as an independent predictive check.
Set `method="split_conformal"` to calibrate width on rows where `holdout`
is zero and evaluate coverage on the remaining rows. Supply both groups and
predictions from a model fitted independently of them. The mode rejects a
confidence level that would require an infinite interval at the available
calibration sample size. Marginal coverage depends on exchangeable scores,
which requires particular care for time series.

Use `innovate.execute` in Python and `kernel_call` in Julia or R.

Voiage owns VOI/economic analysis; Innovate hands off adoption/scenario tables
through Polars/Arrow. Kairos owns event scheduling, DES/ABM state and deterministic
replay. [The migration plan](conductor/tracks/scientific_ecosystem_alignment_20261029/migration-plan.md)
records inspected source revisions, recommendations and remaining gates.

Only Python, Julia and R bindings are maintained. TypeScript, Go and C# packages
were removed; earlier documentation and Conductor records are historical evidence.
Project-owned code is licensed under [Apache-2.0](LICENSE).
