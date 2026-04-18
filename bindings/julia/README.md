# innovate Julia bindings

This package is the Julia-facing adapter over the Python `innovate` functional kernel.

## Invocation path

The Julia scaffold resolves the bridge entrypoint at `inst/python/kernel_bridge.py` and keeps the
runtime boundary thin. The package is designed to shell out to the kernel bridge in later phases
without duplicating model logic in Julia.

## Current scaffold

- Package metadata in `Project.toml`
- Julia module in `src/Innovate.jl`
- Contract and wrapper tests under `test/`
- A Python bridge entrypoint under `inst/python/`
- Stable operations for discovery, fit, predict, simulate, summarize, and diagnose

## Example workflow

```julia
include(joinpath(@__DIR__, "src", "Innovate.jl"))

using .Innovate

request = kernel_request(operation = "discover_models")
schema_version = kernel_schema_version()
models = kernel_discover_models()
```
