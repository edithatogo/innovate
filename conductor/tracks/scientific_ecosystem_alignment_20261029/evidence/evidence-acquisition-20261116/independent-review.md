# Independent review of the repeatable native evidence recipe

Reviewed on 16 November 2026 by a separate automated reviewer. The build helper,
candidate workflow, guard tests and retained execution packet received source
inspection and bounded independent checks. **No blocking finding remains in
this reviewed scope.** The workflow is a track-local proposal; adoption and
hosted execution remain with the #624 integrator.

## Reviewed files

| Path | SHA-256 |
| --- | --- |
| `conductor/tracks/scientific_ecosystem_alignment_20261029/integration/build_scenario_wheel.py` | `61e47e060a5f33bee2120c901b6e6b2f43696ef74da742aef6d5f1a1da233730` |
| `conductor/tracks/scientific_ecosystem_alignment_20261029/integration/scenario-arrow-msrv.yml` | `a67d170cfdd78b8c4ab62e48fc952ba4de82728e80d365575f8ed07c6f9f5f45` |
| `tests/native/test_scenario_wheel_build.py` | `0026de73044f35ecd32187463b7a42fd3321d961fd40c4749b8c707856dcb4d8` |
| Existing `scripts/qualify_scenario_arrow.py` | `fe894db45329c4fd9906aec6a77f8bd87653b0a7d927ac4aeaed39e6e099b3db` |

The helper hash equals the executor hash in the
[actual build record](build-provenance.json). The existing qualifier hash equals
both the build record and [qualification receipt](qualification.json).

## Source and artifact identity

The actual build and qualification record local integration commit
`2748084292e5e57c181c1665a686ad3a291145bf`, tree
`787eb4cd175b7343026cdad88143e8879e0b3062`. Its parents are published PR #662 head
`d36c33f6b4fb480828ddb9aae0121f3379667015` and main
`b29a1d27f54fdd08d77881ee999772fceab7c03a`.

The reviewer independently compared **all 178 recorded build-input hashes** with
both the working files and `git show b29a1d2:path` blobs. Every comparison
matched. The build record and qualification carry the same complete input map.
All three test/fixture input hashes match the checked repository files.
[Current-main verification](current-main-source-verification.json) records the
owner's separate comparison. This establishes equality of the selected inputs,
not equality of the complete integration and main trees.

The retained [commit object](source-commit-object.txt) was independently passed
to `git hash-object -t commit`; it produces exactly the local integration commit
above. Its file SHA-256 is
`6406421754df052abd5a22029bd0c65b95f71c0d71fb85ca0a8ac2cae9b1f5d9`.
This preserves the commit object's identity without representing that local
commit as an already public ref.

The actual wheel file in the recorded build output exists and hashes to
`1c2ea5ef7b6408ba750600c3abbbb4a405b39ad42367abab5dd71d0e3842e93b`.
The reviewer reran the qualifier's wheel/RECORD comparison against the retained
consumer environment: **all four installed package members match**. The
compiled native-module bytes also match the receipt. No wheel rebuild or native
test rerun was needed for this identity check.

## Observed execution evidence

| Check | Independently inspected result |
| --- | --- |
| Development-wheel build | Actual log reports the native crates and successful Maturin wheel output; recorded Rust 1.100.0 and Maturin 1.16.0 |
| Isolated installed consumer | Python 3.15.2; Innovate 0.7.0a1, Polars 2.1.0, PyArrow 26.0.1 and pytest 9.2.0 |
| JUnit and pytest output | 42 distinct passing cases, including all four optional Arrow cases; zero skipped, failed or errored cases |
| Declared-minimum check | Separate Rust 1.99.1 / Cargo 1.99.1 receipt, exit 0; actual log completes the workspace/all-target check |
| Retained identity | Build log, build record, JUnit, pytest log and MSRV log SHA-256 values all match their receipts |

The [installation log](installation.log) records the fresh external environment,
identified wheel installation, dependency check and isolated qualification
command. The [MSRV receipt](msrv-receipt.json) and [log](msrv-check.log) describe
Linux, default features and all five workspace crates. The excluded games crate,
execution at the minimum compiler, other platforms and release acceptance are
outside that compile result.

The installed result is the bounded scenario-export/native-producer
qualification already defined by the existing qualifier. It is not all 38
scientific obligations, a full dependency-closure execution or release
qualification.

## Helper and recipe behavior

The helper uses a fresh output directory and refuses reuse. It checks the exact
Python runtime and manifest-declared Maturin backend, rejects dirty/untracked
native inputs, records source, provider, compiler, command and artifact
identities, and compares native inputs again after building. A nonzero real
builder exit preserves diagnostics and does not emit a successful provenance
record. A missing, ambiguous or symlink wheel is rejected. The existing
qualifier remains responsible for installed wheel identity, exact case
selection, version checks and rejection of skips.

The candidate workflow uses pinned action revisions, explicit Python/Rust
versions, an outside-checkout consumer environment, isolated qualification,
retained artifacts after failure, and `pipefail` for the minimum-compiler
command. Its path filters cover the selected source, build, qualifier and
scenario-test inputs. The workflow has not been installed under
`.github/workflows` or executed on a hosted runner in this packet. The actual
local commands support the component recipe; they do not establish execution of
the proposed entire hosted workflow or its complete `uv sync --locked`
development environment.

## Guard tests and resolved portability point

The reviewer executed:

```sh
/workspace/scratch/e4108ca2b049/innovate-scientific/.venv-registry-reconciliation/bin/python -B -m pytest -p no:cacheprovider tests/native/test_scenario_wheel_build.py
```

Result: **24 passed in 0.43 seconds**, before the narrow test-only portability
adjustment below. This includes an actual subprocess returning exit 19 and
retained diagnostics; the build orchestration tests use explicitly synthetic
test artifacts and do not qualify a native wheel.

The real-subprocess guard originally supplied an empty environment. Because
the native guard suite also runs on Windows, the reviewer recommended preserving
the host environment, as production already does. Python's
[subprocess documentation](https://docs.python.org/3/library/subprocess.html#subprocess.Popen)
requires supplied environments to contain variables needed by the child,
including `SystemRoot` for Windows side-by-side assemblies. This was a
portability risk identified from the test code and documented contract, not an
observed Windows failure.

The owner changed that guard to `os.environ.copy()`, retaining the real exit-19
and exact-log assertions. The reviewer then executed the changed test:

```sh
/workspace/scratch/e4108ca2b049/innovate-scientific/.venv-registry-reconciliation/bin/python -B -m pytest -p no:cacheprovider tests/native/test_scenario_wheel_build.py::test_builder_preserves_real_subprocess_exit_and_diagnostics
```

Result: **1 passed in 0.08 seconds**. The file hashes above identify the final
reviewed version. The production helper and its executed artifact identity were
unchanged by this test-only fix. The owner's combined final guard run is a
separate verification receipt.

This review supplies source and evidence scrutiny. It does not supply scientific
sign-off, a rights declaration, human review, author approval or a release
decision.
