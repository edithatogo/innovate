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
the equivalent IPC stream and the file produced by PyArrow 26.

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

For local bundled-library testing, stage a copy of the Julia package and
place the locked Rust shared library in its `deps` directory under the
name `libinnovate_ffi` with the platform's shared-library extension.
Add the staged package to a fresh project with `Pkg.develop(path=...)`
and unset `INNOVATE_NATIVE_LIBRARY` to exercise bundled discovery.
