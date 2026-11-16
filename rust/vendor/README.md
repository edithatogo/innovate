# Vendored Rust dependencies

These upstream sources preserve the numerical and NUTS dependency versions
while their dependency manifests use `pastey` in place of the unmaintained
`paste` crate. Parent manifests use paths so that a Git or source-distribution
consumer receives the same dependency graph. `gemm-f16` is retained for the
upstream optional feature, even when the current Innovate graph does not enable
it. Upstream licences, notices and original manifests are retained alongside
each crate; `UPSTREAM.json` records the original archive checksums and revisions.

Cargo advisory tools skip path dependencies. `Cargo.lock` in this directory is
therefore an **audit inventory**, not a build lock: it records the original
crates.io identities of these vendored sources so RustSec can monitor them.
The security workflow checks that the inventory, crate manifests and audit
lock agree before auditing it. After reviewing any upstream change, update the
inventory and regenerate the audit lock from the repository root with the
repository-local Python interpreter:

```sh
.venv-native/bin/python -c 'from scripts.check_vendor_audit import VENDOR, audit_lock; (VENDOR / "Cargo.lock").write_text(audit_lock(VENDOR), encoding="utf-8")'
```

`innovate-core` is marked `publish = false`: publishing a Cargo dependency with
both a path and a version would otherwise substitute the crates.io package and
restore `paste` for registry consumers. Remove this block only after the
published dependency graph has been independently verified to be paste-free.
The Python metadata still declares the project-owned Apache-2.0 licence. Before
any distribution release, review the complete bundled dependency licences and
the distribution-level SPDX expression; the retained upstream notices include
MIT, BSD and MPL terms and are not relicensed by the project declaration.
