# Issue #556: explicit scenario outcomes at the Arrow boundary

## Scope and reconciliation

This continuation starts from main
`276c2e505a0ce26910820fe8a94a6a498f505700`. PR #554 is merged, but its
Python 3.15.1 candidate predates the authorised Rust/Polars migration. The
selected current runtime is Python 3.15.2 and Rust/Cargo 1.99.1, with locked
Polars 2.1.0 and optional PyArrow 26.0.1. Provider and dependency pins are
unchanged. The development migration check still reports 38 release obligations.

The public Python and C interfaces accept caller-supplied scenario envelopes.
Previously, indexing absent JSON `result` and `error` members yielded null,
allowing an incomplete execution or comparison to acquire `successful=true`.
Export now requires an explicit object result with null error, or null result
with an object error. A successful outer envelope must explicitly contain
`error: null`. Malformed outcomes fail before any Arrow bytes are returned.
Valid execution, error and paused rows retain their previous semantics; paused
executions remain unsuccessful. Nested JSON is preserved without converting
integer identities to floating point.

This is outcome-shape validation, not validation of every model-specific result
or error field. It does not add a result status enum, impose a new schema-version
requirement, qualify numerical models or verify caller identity.

## Executable red phase

Tests-only remote commit:
`089cea95c043b18fa15ccd770fbd1e4d8e7f0dca`,
[PR #602](https://github.com/edithatogo/innovate/pull/602).

Two local Cargo attempts were killed during dependency compilation: first
`polars-arrow` with two jobs, then `polars-core` with one job. Both exited 101
with rustc terminated by SIGKILL. No tests executed in either attempt. These
resource failures are not behavioural red evidence.

Instead, the retained release dry-run artifact from
[run 42443001407](https://github.com/edithatogo/innovate/actions/runs/42443001407)
provided an executable baseline. It was produced at
`768e9df2be85094e0b210c6600fc46abaec3b644`; it is not a build of the new PR.
Artifact ID: `13602064406`.

| Artifact | SHA-256 |
| --- | --- |
| Downloaded artifact ZIP | `1744cd0dbf9ddac2e9f5afd9ef84e8a92a2f7bf9243561950557a95424a286c9` |
| `innovate-0.7.0a1.tar.gz` | `85f3d7a52c5be98265908b3fe88a8ee373a69f3e69e4585852b1b3621ea09f23` |
| `innovate-0.7.0a1-cp315-cp315-manylinux_2_38_x86_64.whl` | `5d1f139531d0f1c397284dfe918d43021f4a6696ec39d7cae61c402f9f2acc92` |

The ZIP digest matched GitHub's artifact metadata; the wheel and sdist matched
the included `SHA256SUMS`. The following files in the sdist matched the current
pre-fix checkout byte for byte:

| File | SHA-256 |
| --- | --- |
| `rust/crates/innovate-tabular/src/lib.rs` | `157ac71b992b196ba616b3eb16fc53f614b36750814e3696ea444a5172e06e1f` |
| `rust/crates/innovate-python/src/lib.rs` | `bd29e76207ffe9ea8357fe998069606bbf71531e5b12a84dadfbb335ab303e66` |
| `bindings/python/innovate/__init__.py` | `4d93a7d5c8272814f7cfa558c0f911af4c6bd6fdc44c03fe65966c62317a307d` |
| `Cargo.lock` | `42efd1345044b445469ad0c40c8abd0288a7d123a2bd4568c578703567a8db67` |

`uv.lock` is not included in that sdist, so no archived Python-lock equality is
claimed. Python dependencies came from the current unchanged lock:

```sh
uv sync --locked --no-install-project --extra arrow --python 3.15.2
uv pip install --python .venv/bin/python --no-deps <retained-wheel>
uv pip check --python .venv/bin/python
.venv/bin/python -m pytest tests/native/test_scenario_exports.py
```

The interpreter reported CPython 3.15.2 on Linux x86_64. Both `innovate` and
`innovate._native` resolved inside the isolated environment's `site-packages`,
without checkout path injection. Dependency checking passed for 100 packages;
NumPy was absent. The focused suite reported **33 failed, 5 passed in 0.78s**.
All 33 failures were `DID NOT RAISE ValueError` for the invalid inputs. The five
passing controls included valid outcome preservation and all four optional
PyArrow file/stream readback and reverse-container round trips.

This is a retained installed-consumer red run with verified equivalence of the
affected source and native lock, not a new full-repository baseline build.

## Green verification and review

Pending execution against the changed native library. The retained wheel is not
green evidence for the fix. Existing hosted checks remain required at the final
PR head. Ruff, Rust formatting and whitespace checks passed for the tests-only
revision; current implementation checks will be recorded separately.

## Remaining acceptance and ownership

The integration coordinator owns shared CI, provider pins, manifests and main
promotion. PR #600's ensemble time-grid work and PR #590's security draft are
separate. This PR requests an integration verification/merge window after its
final required checks and review pass.

The optional PyArrow cases are executable when the existing `arrow` extra is
installed. Ordinary base CI does not install that extra; a skip is not optional
consumer qualification. A recurring optional consumer lane is a shared-CI
integration dependency, not added by this packet.

Historical ArviZ and pytest-benchmark incompatibilities remain in
`qualification.md`. Their Python numerical routes were retired by the native
migration; this work does not claim upstream compatibility fixes. Released
Arrow.jl direct-file/metadata qualification, broader installed-consumer/provider
acceptance, four historical manual scenario checks and all 38 scientific release
obligations remain open. Issue #556 and this track remain open.
