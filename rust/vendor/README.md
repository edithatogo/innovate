# Vendored Rust dependencies

These upstream sources preserve the numerical and NUTS dependency versions
while their dependency manifests use `pastey` in place of the unmaintained
`paste` crate. Parent manifests use paths so that a Git or source-distribution
consumer receives the same dependency graph. `gemm-f16` is retained for the
upstream optional feature, even when the current Innovate graph does not enable
it. Upstream licences, notices and original manifests are retained alongside
each crate; `UPSTREAM.json` records the original archive checksums and revisions.
In `faer/src/sort.rs`, empty block-sort
cursors initially point to their offset arrays instead of using null sentinels.
The loop already reset each empty cursor to that array before dereferencing it;
the new initialization makes that invariant explicit. `UPSTREAM.json` records
the original and patched file hashes, and the audit check verifies the latter.

The `polars-arrow` and `polars-compute` 0.55.2 sources carry a backport of
the Jiff conversions from [Polars #29110](https://github.com/pola-rs/polars/pull/29110)
at `e3d55b387184bf88cf61bca6acff6d8c02dfd7c5`, adapted to Jiff 1.0.3 and
the published 0.55.2 sources. The workspace patch keeps the existing
`polars-core` release and matches its Arrow and compute crate instances
across the native bindings. It uses Jiff's bundled time-zone data for the
locked Innovate profile. The audit inventory retains both original crate
identities and the hashes of each changed source or manifest. Qualification
applies to Innovate's declared feature set. Requalify against a published
Polars replacement before changing that set or distributing independent Rust
packages.

Cargo advisory tools skip path dependencies. `Cargo.lock` in this directory is
therefore an **audit inventory**, not a build lock: it records the original
crates.io identities of these vendored sources so RustSec can monitor them.
The security workflow checks that the inventory, crate manifests and audit
lock agree before auditing it. The wheel carries `THIRD-PARTY-LICENSES`, which
groups byte-identical notices and identifies every source file they cover;
the source distribution also retains the individual upstream files. The same
check detects when the bundle needs to be refreshed. After reviewing any
upstream change, update the inventory and regenerate both files from the
repository root with the repository-local Python interpreter:

```sh
.venv-native/bin/python -c 'from scripts.check_vendor_audit import ROOT, VENDOR, audit_lock, license_bundle; (VENDOR / "Cargo.lock").write_text(audit_lock(VENDOR), encoding="utf-8"); (ROOT / "THIRD-PARTY-LICENSES").write_bytes(license_bundle(VENDOR))'
```

`innovate-core` is marked `publish = false`: publishing a Cargo dependency with
both a path and a version would otherwise substitute the crates.io package and
restore `paste` for registry consumers. Remove this block only after the
published dependency graph has been independently verified to be paste-free.
The Python metadata still declares the project-owned Apache-2.0 licence. Before
any distribution release, review the complete bundled dependency licences and
the distribution-level SPDX expression; the retained upstream notices include
MIT, BSD and MPL terms and are not relicensed by the project declaration.
