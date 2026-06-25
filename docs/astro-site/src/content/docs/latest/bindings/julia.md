---
title: Julia Bindings
description: Julia package adapter over the shared kernel bridge.
---

The Julia bindings expose the stable `innovate` kernel through a thin adapter
layer. The Julia package does not reimplement model behavior; it shells out to
the shared Python kernel bridge and normalizes the results into Julia-friendly
data structures.

## Installation

For checkout development, instantiate the Julia environment and run the package
tests:

```bash
julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=bindings/julia bindings/julia/test/runtests.jl
```

In a repository checkout the binding uses `uv run python` by default. In an
installed-package context it uses `INNOVATE_PYTHON_COMMAND` when provided, or
`python3` when no override is set. The Julia package only prepends `PYTHONPATH`
with the repository `src/` directory when a checkout is detected.

## Checkout example

```julia
include(joinpath(@__DIR__, "..", "..", "..", "bindings", "julia", "src", "Innovate.jl"))

using .Innovate

discovery = kernel_discover_models()
bass = first(filter(record -> record["key"] == "bass", discovery))

fit = kernel_fit_model(
    kernel_request(
        operation = "fit_model",
        model_key = bass["key"],
        payload = Dict(
            "inputs" => Dict("time" => [0.0, 1.0, 2.0, 3.0], "observed" => [0.02, 0.06, 0.12, 0.25]),
            "model_kwargs" => Dict{String,Any}(),
        ),
    ),
)

diagnostics = kernel_extract_diagnostics(fit)
```

The same pattern works for `predict_model`, `simulate_model`, `summarize_model`,
and `diagnose_model`.

Installed-package usage is validated by the smoke script in
`bindings/julia/test/installed_package_smoke.jl`.

## Compatibility and drift checks

The Julia package keeps its schema version aligned with the Python kernel
contract. The automated test suite checks that the Julia and Python schema
version constants match, and it also exercises the end-to-end example in
`bindings/julia/examples/end_to_end.jl` during package tests. A separate
installed-package smoke script is used for registry readiness.

## Support boundaries

- The Julia layer remains thin and contract-driven.
- Only the stable kernel operations are wrapped.
- The package supports Julia General registry installation, with
  installed-package smoke validation in CI and publish gates. For the monorepo
  layout, use Registrator with `subdir=bindings/julia` and expect manual review
  unless a dedicated `Innovate.jl` repository is used.
- Future Arrow-based interchange work will extend the same contract boundary
  rather than replacing it with Julia-native model logic.
