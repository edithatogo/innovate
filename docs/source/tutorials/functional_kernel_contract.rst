Functional Kernel Contract
==========================

The functional kernel is the language-neutral execution surface for ``innovate``.
It keeps the public contract focused on serializable envelopes and Arrow-friendly
payloads so downstream bindings can call into the library without depending on
Python object identity or backend-specific internals.

Contract overview
-----------------

The kernel publishes a versioned schema through ``innovate.kernel.KERNEL_SCHEMA_VERSION``
and a fixed operation list through ``innovate.kernel.KERNEL_OPERATIONS``. The current
operations are:

* ``discover_models``
* ``fit_model``
* ``predict_model``
* ``simulate_model``
* ``summarize_model``
* ``diagnose_model``

Request and response envelopes are represented by:

* ``innovate.kernel.KernelRequest``
* ``innovate.kernel.KernelResponse``
* ``innovate.kernel.KernelError``

Portable payloads
------------------

Where tabular or array-shaped data is exchanged, the kernel uses explicit payload
objects instead of exposing NumPy arrays directly:

* ``innovate.kernel.KernelArrayPayload`` for numeric array data
* ``innovate.kernel.KernelTablePayload`` for row/column tabular data
* ``innovate.kernel.KernelDiscoveryResponse`` for machine-readable model discovery

These payloads serialize to JSON-friendly dictionaries and are intended to map cleanly
to Arrow tables or language-native equivalents in future bindings.

Relationship to the OO API
--------------------------

The kernel complements the Python object-oriented API rather than replacing it.
The object model remains the ergonomic layer for interactive Python workflows,
while the kernel defines the stable cross-language contract that bindings can target
directly. Discovery is already wired to the canonical capability registry; execution
adapters are added in later tracks.
