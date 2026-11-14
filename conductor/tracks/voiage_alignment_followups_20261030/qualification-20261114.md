# Normalized Voiage consumer qualification

This bounded continuation addresses [issue #559](https://github.com/edithatogo/innovate/issues/559)
inside the existing follow-up track. It starts from Innovate
`276c2e505a0ce26910820fe8a94a6a498f505700`. The parent scientific programme,
38 scientific obligations and release restrictions remain in force.

## Implemented boundary

The existing `scripts/qualify_voiage_arrow.py` now uses
`voiage.contracts.normalized_input.NormalizedInputBundle` for IPC and Parquet
reads. It compares the producer's separately supplied `content_digest` **before**
converting to Polars. It does not reimplement Voiage metadata decoding or add
Voiage to Innovate's runtime dependency closure.

The qualification covers simple tables, multiple chunks, timestamp-second
normalization, custom nested covariance field names/units, and schema metadata.
The full original Arrow schema and content digest must survive persistence.
After verification, the numerical consumer explicitly selects `time` and
`sample_id`; it exercises the existing native Logistic implementation against
the independently determined values `[1, 2, 3]`. Timestamp and covariance
preservation do not claim additional native datetime or covariance semantics.
Existing native Bass, Arrow payoff and weighted/unweighted EVPI checks remain.

The command-line launcher starts a fresh `python -I` worker in a temporary
working directory outside the checkout. Installed package paths must be inside
that interpreter's environment and outside the source checkout. Supplied wheels
must contain the actually imported initializers, native extensions and normalized
contract module; all their package members must match installed bytes. A failed
worker preserves its nonzero status and emits a failure receipt.

Receipts include the worker command, exit status, working directory, package
versions, imported paths, wheel/RECORD/native-module hashes, qualification source
revision and script hash, original/restored bundle digests, input file hashes,
explicit native projection and output IPC hashes. The qualification source
revision identifies the checked-out harness. Wheel build provenance must also
be read from its build command/run; the harness does not infer a wheel's source
commit from a version string.

## Upstream identity inspected

- Current source inspected: Voiage
  [`8a3ef4f89d3d630425ace05450e957cfec1bb39c`](https://github.com/edithatogo/voiage/tree/8a3ef4f89d3d630425ace05450e957cfec1bb39c).
- The signed `v3.1.0` tag object `0540cf79c4dae77144ab25a08a650c8967a506f7`
  points to `d40af0276019a7d63657349b53dad96845e2b4c1`.
- The installed Voiage 3.1.0 normalized-input module matches the inspected
  current source blob `28d5d233b71301252bb971054a5e1e4dfcdcf817` byte for byte.
  Its SHA-256 is
  `846dbdf6aee4ec843a7147ba6ba42e3ad8618fcb6c88755332b341150e61b08c`.
  This is a module-level comparison, not an assertion that the whole wheel
  was built from current main.
- Referenced upstream PRs [#1220](https://github.com/edithatogo/voiage/pull/1220)
  and [#1239](https://github.com/edithatogo/voiage/pull/1239) are already merged.

## Executed local checks

Actual interpreter: Python **3.15.2**. Installed contract environment:
Voiage **3.1.0**, Polars **2.1.0**, PyArrow **26.0.1**, pytest **9.2.0**.

```sh
python -I -m pytest tests/consumer/test_voiage_normalized_input.py -q
```

Result: **19 passed**, exit **0**. These are real Voiage/Arrow/Polars contract,
artifact-verification and launcher checks. No dependency stub or missing-package
skip substitutes for them. They do not by themselves qualify Innovate's native
numerical execution.

Observed test-first evidence: the first 15 cases failed because
`read_normalized_frame` was absent, after separating the existing native import
from the independently executable boundary helpers. The three added wheel
identity cases separately failed because `verify_installed_wheel` was absent.
Both sets passed after implementation. The initial collection failure caused by
the missing local native wheel is not counted as a behavioural regression test.

Ruff **0.17.4** lint and formatting passed for the changed Python files.
Actionlint **1.17.1** (including ShellCheck), Zizmor **1.32.0** normal offline
audit, Conductor registry hygiene and `git diff --check` passed. Independent
review identified and corrected the unsupported native timestamp projection and
the missing-required-wheel-member gap before hosted submission.

## Native installed-artifact gate

Local Innovate compilation was terminated by signal 9: initially during an
unbounded editable build, then during `polars-core` compilation with two Cargo
jobs and again with one. No source/compiler diagnostic established a code
defect. These builds failed; no locally installed Innovate numerical pass is
claimed.

The new `.github/workflows/voiage-consumer.yml` reuses
`scripts/build_release_wheel.sh` to build the native artifact from its source
distribution on a hosted runner. It installs explicit Innovate and Voiage wheel
files outside the checkout, runs the focused tests there, runs the existing
qualification harness, and retains its evidence even on failure. Permissions are
read-only; it publishes no package and changes no existing required gate.
Hosted qualification is pending at this source checkpoint.

The equivalent installed invocation is:

```sh
/path/to/consumer/bin/python -I scripts/qualify_voiage_arrow.py \
  --wheel /absolute/path/to/innovate.whl \
  --voiage-wheel /absolute/path/to/voiage.whl
```

## Remaining acceptance

See [dependency and automation audit](alignment-audit-20261114.md) for the
basedpyright/Presidio exceptions, the live Dependabot and signing visibility
limits, and the unexecuted strict Renovate validation. Source configuration and
contract tests do not close those checks. Retain issue #559 and this track until
installed native evidence, exact-head hosted gates and the outstanding security
acceptance are reconciled by their owners. No release or broader-track
completion follows from this increment.
