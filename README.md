# Innovate

Innovate models innovation adoption, technology substitution and policy diffusion.
The 0.7 development line uses a **Rust scientific core**, **Polars** tables and
thin **Python, Julia and R** bindings. This is a breaking migration in progress.
All current and planned scientific features remain required; unported features
are temporarily unavailable and return explicit errors. There is no Python
numerical fallback. Rust is the core library, not a fourth binding.

## Current native scope

The kernel implements bounded parameter-only Bass, Logistic, Gompertz,
Fisher–Pry and single-generation Norton–Bass fit/predict/simulate/summary paths.
These are development capabilities, not full scientific or release acceptance.
Advanced model families, covariates, event splits, uncertainty, native Kairos
simulation and the remaining areas in [the migration ledger](migration-ledger.json)
still require implementation or qualification. The release gate currently fails.
The earlier Python suite remains in Git history as migration evidence; the new
native suite does not claim equivalent feature coverage.

Polars DataFrames/LazyFrames are the Python table interface. Rust reads/writes
Arrow IPC and owns table validation and predictions. IPC currently copies buffers;
LazyFrames collect at the native execution boundary. R/Julia currently expose
the versioned JSON contract through a native C ABI. Their tabular IPC adapters
and distributable native-library bundles remain open.

## Development

Use Rust/Cargo 1.99.1 and repo-local Python 3.15.2. Keep any earlier environment
until its active operations finish. The new environment is separate:

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
recovery is tested, but general identifiability and calibrated uncertainty remain open.

Voiage owns VOI/economic analysis; Innovate hands off adoption/scenario tables
through Polars/Arrow. Kairos owns event scheduling, DES/ABM state and deterministic
replay. [The migration plan](conductor/tracks/scientific_ecosystem_alignment_20261029/migration-plan.md)
records inspected source revisions, recommendations and remaining gates.

Only Python, Julia and R bindings are maintained. TypeScript, Go and C# packages
were removed; earlier documentation and Conductor records are historical evidence.
Project-owned code is licensed under [Apache-2.0](LICENSE).
