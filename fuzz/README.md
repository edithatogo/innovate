# Kernel request fuzzing

The existing `kernel_request` target exercises `innovate_core::execute_json`
using synthetic bytes. The campaign uses Rust `nightly-2026-11-19` and
`cargo-fuzz 0.13.2`; compiler upgrades should be reviewed separately from corpus
changes. The native application toolchain remains unchanged.

The isolated fuzz lockfile is refreshed to match the current native manifests.
CI fetches it with `--locked`, builds offline, and rejects any lockfile change
before executing the campaign. The log records its SHA-256. This supplies the
lock check that cargo-fuzz 0.13.2 does not expose as a command-line flag.

The workflow runs for 30 seconds on relevant pull requests and 300 seconds on
weekly/manual runs. Inputs are capped at 64 KiB, the fuzzer RSS limit is 1 GiB,
and any one input has a five-second timeout. The overall job limit is 20 minutes.
Sixteen code generation units reduce build pressure while retaining optimized
code, debug assertions, overflow checks and the default address sanitizer.

```sh
cargo +nightly-2026-11-19 install cargo-fuzz --version 0.13.2 --locked
cargo +nightly-2026-11-19 fetch --locked --manifest-path fuzz/Cargo.toml
cargo +nightly-2026-11-19 fuzz run kernel_request --codegen-units 16 -- -max_total_time=30 -max_len=65536 -rss_limit_mb=1024 -timeout=5
```

CI retains the actual source revision, compiler version, campaign log and any
failure reproducers for 14 days. Replay a minimized artifact at the same source
and toolchain with:

```sh
cargo +nightly-2026-11-19 fuzz run kernel_request --codegen-units 16 fuzz/artifacts/kernel_request/<artifact>
```

After a confirmed defect is fixed, keep its minimized synthetic input as a
regression seed. Do not add credentials, private research records or live source
responses. Compilation alone does not establish execution; a successful bounded
campaign does not establish complete memory safety, scientific validity or the
broader Q01/FFI/release acceptance gates.
