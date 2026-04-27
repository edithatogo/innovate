C# Bindings
===========

The C# bindings are a planned thin adapter over the stable ``innovate`` kernel
contract. The C# layer should not reimplement model behavior; it should build
typed request objects, invoke the shared kernel bridge, and convert responses
into .NET-friendly structures.

Planned package shape
---------------------

The initial package should use a conventional .NET 11 SDK project layout:

.. code-block:: text

   bindings/csharp/
   ├── Innovate.Kernel/
   │   ├── Innovate.Kernel.csproj
   │   ├── KernelRequest.cs
   │   ├── KernelResponse.cs
   │   ├── KernelBinding.cs
   │   └── KernelError.cs
   ├── Innovate.Kernel.Tests/
   │   ├── Innovate.Kernel.Tests.csproj
   │   ├── SchemaCompatibilityTests.cs
   │   ├── KernelContractTests.cs
   │   └── ErrorMappingTests.cs
   └── README.md

The binding should use ``INNOVATE_PYTHON_COMMAND`` to select the Python launcher,
matching the existing language bindings. The default command should remain
``uv`` with an invocation equivalent to ``uv run python`` from the repository
root.

Schema mapping
--------------

The C# layer should map the kernel contract into explicit types:

* ``KernelRequest`` maps to the shared request envelope, including
  ``schema_version``, ``operation``, ``model_key``, ``payload``, and
  ``metadata``.
* ``KernelResponse`` maps to the shared response envelope, including
  ``schema_version``, ``operation``, ``result``, ``error``, and ``metadata``.
* ``KernelError`` maps to the stable error payload and preserves the kernel
  error code, message, operation, details, and retryability flag.

The first supported operation should be ``discover_models`` because it is
metadata-driven and does not require fitted model state. Follow-up wrappers may
cover ``fit_model``, ``predict_model``, ``simulate_model``, ``summarize_model``,
and ``diagnose_model`` once schema compatibility checks are in place.

Compatibility and drift checks
------------------------------

The C# package should include tests that verify:

* schema compatibility with the Python kernel ``KERNEL_SCHEMA_VERSION``
* operation-name compatibility with the stable kernel operation list
* request serialization for ``KernelRequest``
* response deserialization for ``KernelResponse``
* error mapping for stable kernel error codes
* an end-to-end ``discover_models`` smoke test through the shared bridge

Support boundaries
------------------

* The C# layer is planned and should be treated as provisional until tests and
  package scaffolding exist.
* The binding should remain a thin adapter over the shared kernel contract.
* It does not reimplement model behavior.
* It should not become a separate public API or execution core.
* Future transport, Arrow, or FFI work should extend the same contract boundary
  used by the R, Rust, Julia, TypeScript, and Go bindings.
