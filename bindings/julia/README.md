# innovate Julia bindings

This package is the Julia-facing adapter over the Python `innovate` functional kernel.

## Installation

For checkout development:

```bash
julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=bindings/julia bindings/julia/test/runtests.jl
```

## Invocation path

The Julia package resolves the bridge entrypoint at `inst/python/kernel_bridge.py` and keeps the
runtime boundary thin. The package shells out to the Python kernel bridge instead of duplicating
model logic in Julia.

## Current scaffold

- Package metadata in `Project.toml`
- Julia module in `src/Innovate.jl`
- Contract and wrapper tests under `test/`
- A Python bridge entrypoint under `inst/python/`
- Stable operations for discovery, fit, predict, simulate, summarize, and diagnose
- Schema-compatibility drift checks in `test/schema_compatibility.jl`

## Backend expectations

- Julia 1.12 or newer is required.
- In a repository checkout the bridge uses `uv run python` by default.
- In an installed-package context the bridge uses the configured Python launcher from
  `INNOVATE_PYTHON_COMMAND`, or `python3` if no override is set.
- The Julia package only prepends `PYTHONPATH` with the repository `src/` directory when a
  checkout is detected.

## Checkout example

```julia
include(joinpath(@__DIR__, "src", "Innovate.jl"))

using .Innovate

request = kernel_request(operation = "discover_models")
schema_version = kernel_schema_version()
models = kernel_discover_models()
```

Installed-package usage is validated by the smoke test in `test/installed_package_smoke.jl`.

## Compatibility

The Julia bindings are intentionally thin and contract-driven. The Julia schema version is kept in
lockstep with the Python kernel version, and the automated test suite fails if the versions drift.

## Registry readiness

Julia General registration from this monorepo uses:

```text
@JuliaRegistrator register subdir=bindings/julia
```

The package includes local license metadata and dependency compatibility bounds. Registry readiness
is validated by an installed-package smoke test that can run the bridge against a configured Python
environment with the Python `innovate` package available.
