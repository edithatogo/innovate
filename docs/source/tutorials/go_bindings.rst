Go Bindings
===========

The Go bindings expose the stable `innovate` kernel through a thin adapter
layer. The Go package does not reimplement model behavior; it shells out to the
shared Python kernel bridge and normalizes the results into Go-friendly data
structures.

Installation
------------

From the repository root, run the Go package tests from the module directory:

.. code-block:: bash

   cd bindings/go
   go test ./...

The binding uses ``uv run python`` by default. If your environment requires a
different launcher, set the ``INNOVATE_PYTHON_COMMAND`` environment variable
before calling into the package.

Basic usage
-----------

.. code-block:: go

   package main

   import (
       "fmt"

       innovate "github.com/edithatogo/innovate/bindings/go"
   )

   func main() {
       discovery, err := innovate.DiscoverModels()
       if err != nil {
           panic(err)
       }

       var bass innovate.KernelDiscoveryRecord
       for _, record := range discovery.Models {
           if record.Key == "bass" {
               bass = record
               break
           }
       }
       if bass.Key == "" {
           panic("bass model must be discoverable")
       }

       fit, err := innovate.FitModel(innovate.KernelRequest{
           SchemaVersion: innovate.KernelSchemaVersion(),
           Operation:     "fit_model",
           ModelKey:      &bass.Key,
           Payload: map[string]any{
               "inputs": map[string]any{
                   "time":     []float64{0, 1, 2, 3, 4},
                   "observed": []float64{0.02, 0.06, 0.12, 0.25, 0.41},
               },
               "model_kwargs": map[string]any{},
           },
           Metadata: map[string]any{},
       })
       if err != nil {
           panic(err)
       }

       diagnostics, ok := innovate.ExtractDiagnostics(fit)
       if !ok {
           panic("fit response should expose diagnostics")
       }

       fmt.Println(discovery.SchemaVersion)
       fmt.Println(bass.Key)
       fmt.Println(diagnostics["support_level"])
   }

Compatibility and drift checks
------------------------------

The Go package keeps its schema version aligned with the Python kernel
contract. The automated test suite checks that:

* the discovery response schema version matches ``KernelSchemaVersion()``
* the exported stable operation list matches the wrapper surface
* the end-to-end example in ``bindings/go/example_test.go`` still runs during
  package tests

Support boundaries
------------------

* The Go layer remains thin and contract-driven.
* Only the stable kernel operations are wrapped.
* The package is consumed as a Go module using the import path
  ``github.com/edithatogo/innovate/bindings/go`` and versioned submodule tags
  such as ``bindings/go/v0.5.0``.
* Future transport or Arrow work should extend the same contract boundary
  rather than replacing it with Go-native model logic.
