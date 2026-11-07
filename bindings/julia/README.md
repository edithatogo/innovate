# Innovate JULIA native binding

This 0.7 development package invokes the Rust C ABI, without Python. Build the
core with `cargo build --locked -p innovate-ffi` from the repository root and set
`INNOVATE_NATIVE_LIBRARY` to the absolute shared-library path under `target/debug`.
Only bounded native capabilities are currently exposed. Advanced features and
installed native distribution bundles remain open in the root migration ledger.
The JSON request schema is 2.0. Inputs are borrowed for the call; the wrapper
copies response text and releases the Rust-owned result exactly once.

## Reading causal Arrow exports

Use an Arrow IPC stream when handing a Python Polars causal result to Julia.
For example, given the result of a native `policy_evidence` request in Python:

```python
import polars as pl
from innovate import policy_metadata_frame

policy_metadata_frame(result).write_ipc_stream(
    "policy.arrow", compression="uncompressed", compat_level=pl.CompatLevel.oldest()
)
```

In a Julia consumer environment with Arrow.jl installed:

```julia
using Arrow
metadata = Arrow.Table("policy.arrow")
```

The same stream settings work with `treatment_contrast_frame`. Arrow.jl
2.8.1 returned no columns for the tested Polars IPC file format. It read
the equivalent IPC stream and the file produced by PyArrow 26. With
Python Polars 2.1.0, the empty result reproduced at both compatibility
levels and with both default and one-row batches. Check the expected
column names after opening a file because the reader can return an
empty table without raising an error.

For a file handoff, use Innovate's Polars-backed IPC writer. It emits the
leading schema message that Arrow.jl 2.8.1 expects.

```python
from pathlib import Path
from innovate import export_ipc_file, policy_metadata_frame

Path("policy.arrow").write_bytes(export_ipc_file(policy_metadata_frame(result)))
```

For an existing file emitted by Polars, pass its bytes through the same
exporter. The native reader accepts IPC files and streams.

```python
Path("policy-julia.arrow").write_bytes(export_ipc_file(Path("policy-polars.arrow").read_bytes()))
```

Julia can also rewrite a file directly through Innovate.

```julia
using Innovate
write("policy-julia.arrow", export_ipc_file(read("policy-polars.arrow")))
```

Use the native stream exporter when Julia needs custom schema metadata.
Arrow.jl 2.8.1 reads the leading schema. The file exporter stores custom
schema metadata in the footer, while the stream exporter puts it in the
leading message. Both exporters preserve schema-level metadata.
Column-level annotations such as field units survive when the Arrow
field type stays the same. For fields whose type changes during
compatibility conversion, carry required annotations as explicit
data or schema-level metadata.

```julia
using Innovate, Arrow

table = Arrow.Table(export_ipc_stream(read("policy-julia.arrow")))
metadata = Arrow.getmetadata(table)
```

For local bundled-library testing, stage a copy of the Julia package and
place the locked Rust shared library in its `deps` directory under the
name produced by Cargo, such as `libinnovate_ffi.so` or `innovate_ffi.dll`.
Add the staged package to a fresh project with `Pkg.develop(path=...)`
and unset `INNOVATE_NATIVE_LIBRARY` to exercise bundled discovery.

## Reading paired Bass outcomes

`kernel_paired_outcomes_ipc(payload)` validates paired Bass draws and
returns an Arrow IPC file as bytes. Read that file directly with
`Arrow.Table`. Rows follow strategy, draw and time order. The
`perspective` column is nullable and `parameters_json` retains each
draw's parameter array with names, values and units.

Direct readback was verified with Arrow.jl 2.8.1 on the 24-row historical
diffusion fixture. The stream guidance above applies to files written
directly by Python Polars.
