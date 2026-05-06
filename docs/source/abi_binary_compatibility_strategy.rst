ABI and Binary Compatibility Strategy
=====================================

This strategy defines how Innovate can add native implementations and packaged
binary artifacts without changing the public API. It is intentionally
API-preserving: callers should keep depending on the Python functional kernel,
schema-versioned payloads, binding wrappers, and capability-discovery metadata
rather than on private native implementation details.

Compatibility boundaries
------------------------

Public API compatibility
   The public API is the documented Python package surface, the functional
   kernel operations, the language binding wrappers, and stable error and
   diagnostics payloads. API-preserving ABI changes must not require callers to
   import Rust structs, import private Python modules, or change kernel
   operation names. API-preserving ABI changes must not require callers to link
   against private native symbols.

Kernel schema compatibility
   Kernel request and response compatibility is governed by
   schema-versioned kernel request and response payloads. Schema evolution must
   be additive within a compatible release unless a major-version migration is
   explicitly documented. Language bindings should validate schema versions and
   operation names before they rely on native execution.

Native ABI compatibility
   Native ABI compatibility applies only where Innovate deliberately exposes a
   binary boundary for process, language, or package-manager interoperability.
   Rust private structs are not public ABI. Python objects are not public ABI.
   C++ symbols, JAX objects, jaxlib implementation details, and internal Rust
   module layouts are not public ABI.

Backend capability metadata
   Native implementations must stay behind capability-discovery metadata. A
   backend can report that a model slice is Rust-native, Python-backed,
   XLA-backed, or unsupported, but the reported capability does not expose that
   backend's private ABI.

Arrow native boundary
---------------------

The Arrow C Data Interface is the preferred FFI boundary for tabular arrays,
columnar buffers, and cross-language data movement where binary compatibility is
required. The Arrow C Stream Interface should be used for streaming record
batches or larger table-like payloads.

The ABI promise at this boundary is the Arrow schema and array metadata plus the
documented ownership and lifetime rules from Arrow. Innovate should not invent
ad hoc binary layouts for model states, diagnostics, or intermediate arrays when
Arrow can represent the data. Model-specific Python objects, Rust structs, and
NumPy/JAX array wrapper internals remain implementation details.

XLA and accelerator boundary
----------------------------

XLA internals are not public ABI. This includes jaxlib objects, HLO text,
StableHLO or backend-specific lowering artifacts unless a separate standard is
explicitly adopted, compiled executable handles, device buffer layout, sharding
metadata, and accelerator memory placement details.

JAX/XLA remains an optional accelerator backend. A release can expose that a
kernel slice is XLA-eligible, XLA-backed, or rejected for XLA through
capability-discovery metadata and benchmark dossiers. The compiled artifact is a
capability-gated implementation detail, not a stable interchange format or
package-manager ABI.

Package-manager binary compatibility
------------------------------------

Package-manager artifacts must document the binary promises they actually make:

* PyPI wheels should use explicit Python version, platform, and ABI tags. Native
  wheels should document manylinux, musllinux, macOS universal2, and Windows
  wheel tags when those artifacts are produced.
* conda packages should document run exports or compatible dependency pins for
  native libraries that cross process or language boundaries.
* crates.io artifacts should keep Rust crate semver separate from any public C
  ABI. Rust structs are private unless a future C header explicitly stabilizes
  them.
* npm packages should publish JavaScript and TypeScript contracts, not native
  binary compatibility, unless a future native add-on is introduced with
  explicit Node ABI support.
* CRAN and R-universe packages should document whether compiled code is present
  and which external system libraries are linked.
* Julia General packages should treat the Julia wrapper API and kernel schema as
  stable, while Python bridge and native backend selection remain runtime
  configuration.
* Go modules should use semantic import versioning for API changes. cgo or C ABI
  exposure requires a separate compatibility statement.
* NuGet packages should document target frameworks, RID-specific native assets,
  and whether any native library is part of the public compatibility surface.

Versioning and promotion rules
------------------------------

Native implementation changes can ship in a minor or patch release when they are
API-preserving, keep schema compatibility, and remain behind capability gates.
A release must bump the schema version or major package version when callers are
required to change request payloads, response payloads, binding APIs, or a
published binary ABI.

Before a native or accelerator backend becomes a default, the promotion record
must identify the public API, kernel schema, native ABI boundary if any, package
artifacts affected, fallback behavior, and benchmark evidence. XLA internals,
Rust private structs, and Python object layouts must not be used as compatibility
evidence.
