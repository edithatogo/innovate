# innovate Julia bindings

This package is the Julia-facing adapter over the Python `innovate` functional kernel.

## Installation

From the repository root:

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
- The bridge uses `uv run python` by default. Set `INNOVATE_PYTHON_COMMAND` if you need a
  different Python launcher.
- The Julia package automatically sets `PYTHONPATH` to the repository `src/` directory before
  calling the shared kernel.

## Example workflow

```julia
include(joinpath(@__DIR__, "src", "Innovate.jl"))

using .Innovate

request = kernel_request(operation = "discover_models")
schema_version = kernel_schema_version()
models = kernel_discover_models()
```

## Compatibility

The Julia bindings are intentionally thin and contract-driven. The Julia schema version is kept in
lockstep with the Python kernel version, and the automated test suite fails if the versions drift.
