Julia Bindings
==============

The Julia bindings expose the stable `innovate` kernel through a thin adapter layer. The Julia
package does not reimplement model behavior; it shells out to the shared Python kernel bridge and
normalizes the results into Julia-friendly data structures.

Installation
------------

From the repository root, instantiate the Julia environment and run the package tests:

.. code-block:: bash

   julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate()'
   julia --project=bindings/julia bindings/julia/test/runtests.jl

The binding uses `uv run python` by default. If your environment requires a different launcher, set
the `INNOVATE_PYTHON_COMMAND` environment variable before calling into the package.

Basic usage
-----------

.. code-block:: julia

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

The same pattern works for `predict_model`, `simulate_model`, `summarize_model`, and
`diagnose_model`.

Compatibility and drift checks
------------------------------

The Julia package keeps its schema version aligned with the Python kernel contract. The automated
test suite checks that the Julia and Python schema version constants match, and it also exercises
the end-to-end example in `bindings/julia/examples/end_to_end.jl` during package tests.

Support boundaries
------------------

- The Julia layer remains thin and contract-driven.
- Only the stable kernel operations are wrapped.
- The package is intended for local development and direct repository use, not registry publication.
- Future Arrow-based interchange work will extend the same contract boundary rather than replacing
  it with Julia-native model logic.
