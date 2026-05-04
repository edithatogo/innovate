C# Bindings
===========

The C# bindings are a provisional thin adapter over the stable ``innovate`` kernel
contract. The C# layer does not reimplement model behavior; it builds
typed request objects, invokes the shared kernel bridge, and converts responses
into .NET-friendly structures.

Package shape
-------------

The initial package should use a conventional multi-targeted .NET 10 and
.NET 11 SDK project layout:

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
   │   └── KernelBridgeTests.cs
   └── README.md

The binding should use ``INNOVATE_PYTHON_COMMAND`` to select the Python launcher,
matching the existing language bindings. The default command should remain
``uv`` with an invocation equivalent to ``uv run python`` from the repository
root.

NuGet publication
-----------------

The ``Innovate.Kernel`` project publishes as ``innovate.cs`` for ``net10.0`` and
``net11.0`` and ships as a conventional NuGet package with Apache-2.0 license metadata, repository
metadata, project URL, package tags, release notes, SourceLink settings, symbol
package output, and the package readme. The package also includes the Python
bridge script as ``contentFiles/any/any/innovate/kernel_bridge.py`` so package
artifact checks can prove the bridge content is present before NuGet
publication.

The release workflow must test both target frameworks, create the ``.nupkg`` and
``.snupkg`` artifacts, inspect the generated NuGet metadata, and verify the
readme and bridge script are present before running ``dotnet nuget push``.

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

The first supported operation is ``discover_models`` because it is
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

* The C# layer is provisional until broader wrapper coverage and package
  consumer feedback stabilize.
* The binding should remain a thin adapter over the shared kernel contract.
* It does not reimplement model behavior.
* It should not become a separate public API or execution core.
* Future transport, Arrow, or FFI work should extend the same contract boundary
  used by the R, Rust, Julia, TypeScript, and Go bindings.
