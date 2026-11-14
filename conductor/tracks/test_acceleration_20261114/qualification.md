# Issue #473 qualification

## Reconciled scope

Implementation starts at `276c2e505a0ce26910820fe8a94a6a498f505700`. The issue's
August Python-first inventory is historical. Current production computation is
Rust; Python 3.15.2 and the active native/policy tests are the relevant interface.
PRs #600, #602, #603, #604, #605 and #606 were inspected for ownership. The only
shared-file overlap with #606 is the independent Conductor registry entry; keep
both entries during integration. No existing workflow, production dependency,
native implementation, scientific tolerance or release gate is changed.

## Executed ordinary checks

Selected interpreter: actual CPython 3.15.2 on Linux x86_64. Packages: pytest
9.2.0, Coverage 7.17.2, pytest-testmon 2.3.0 and pytest-gremlins 1.13.2. Optional
tools have a separate universal hash-locked requirements file.

```sh
.venv/bin/python -m pytest \
  tests/native/test_migration_policy.py \
  tests/native/test_migration_mutation.py \
  tests/native/test_test_acceleration.py \
  tests/native/test_test_acceleration_metrics.py \
  tests/native/test_test_acceleration_process.py
.venv/bin/python -m pytest tests/test_acceleration/test_testmon_integration.py
.venv/bin/ruff check scripts/test_acceleration/*.py \
  tests/native/test_test_acceleration*.py tests/native/test_migration_mutation.py \
  tests/test_acceleration/test_testmon_integration.py
.venv/bin/ruff format --check scripts/test_acceleration/*.py \
  tests/native/test_test_acceleration*.py tests/native/test_migration_mutation.py \
  tests/test_acceleration/test_testmon_integration.py
.venv/bin/python scripts/conductor_registry_hygiene.py --json
git diff --check
```

Results: **129 ordinary tests passed**, plus **one opt-in integration test**
covering ten actual testmon scenarios. Ruff, formatting and Conductor hygiene
passed. The 24 initial runner regressions failed against the empty implementation;
the metrics subprocess test also failed before its plugin existed. These are
tooling implementation red/green results, not scientific numerical regressions.

The final combined command executed **130 tests in 9.00 seconds**. A source
archive was built with `maturin sdist`; inspection found no profile cache files.
The existing source-archive inclusion policy leaves these development tools and
their tests in the repository rather than the distribution. No wheel installation
or complete native package acceptance is inferred from constructing that archive.

Combined coverage of the new tooling includes the ordinary tests' real child
processes and parent CLI execution of all five profiles. It measured **288/318
statements (90.57%)** and **89/108 branches (82.41%)**. The metrics plugin measured
100% for both. Separate parent and subprocess coverage configurations avoid
injecting coverage instrumentation into testmon or mutation children. Coverage-
instrumented runs are excluded from the performance samples below.

The new 65 normal-import policy tests are in `tests/native` and therefore execute
in existing ordinary CI without installing Gremlins. The testmon integration test
is outside the default path and explicitly requires optional tools; it cannot
substitute for CI. The two real process-tree tests exercise POSIX behavior and
skip on Windows; Windows process-tree cleanup remains locally unqualified.

## Before/after policy measurements

Reproduce with `python -m scripts.test_acceleration PROFILE --scope policy`.
The machine-readable [evidence](evidence.json) retains every sample, observed
outcome, source hash and the common input snapshot. Three samples were taken for
baseline, warm selection and each coverage core; cold seed and mutation have one
sample. No native extension was installed for this Python-only scope.

| Profile | Collected / executed | Pytest subprocess seconds | Wrapper seconds | Whole command seconds |
| --- | ---: | ---: | ---: | ---: |
| Ordinary policy baseline, median of 3 | 81 / 81 | 0.415 | 0.785 | 0.831 |
| Cold testmon seed | 81 / 81 | 0.642 | 1.076 | 1.118 |
| Warm unchanged testmon, median of 3 | 81 / 0 | 0.294 | 0.704 | 0.744 |
| Coverage ctrace, median of 3 | 81 / 81 | 0.501 | 0.946 | 0.984 |
| Coverage sysmon, median of 3 | 81 / 81 | 0.425 | 0.876 | 0.921 |
| Bounded Gremlins trial | 65 / 65 ordinary baseline; 34 mutants | 8.083 | 8.492 | 8.533 |

Both policy files are explicit pytest inputs, so warm testmon still collects 81
items and deselects all 81. Each warm receipt says `no_tests_selected`: **zero new
tests passed**. The larger difference in subprocess time is reduced by inventory
and hashing overhead. These short local samples support keeping the profiles
optional; they do not establish general native-suite performance gains.

### Actual changed-code sensitivity

The independent synthetic qualification used two unrelated binding functions.
Baseline and seed ran both tests; an unchanged warm run collected/executed zero.
Changing one binding function selected **one of two tests**, deselected the
unaffected test and detected its failure. Restoring the function then reran both
tests because the failed run had invalidated the success stamp. Native-source,
JSON-fixture and manifest changes each caused full reruns; adding a test rebuilt
the inventory and ran all three. A final unchanged run executed zero.

The portable reproducer is
`tests/test_acceleration/test_testmon_integration.py`. It checks real plugin
behavior, environment/version guards and recovery without compiling synthetic
Rust. Source changes invalidate cache reuse; they do not establish that an
installed native binary has been rebuilt from that source.

## Coverage core compatibility

Both profiles explicitly enabled branch measurement and verified the actual
core using Coverage's `--debug=core` output. They measured the same
`scripts/check_migration.py` bytes and the same 81 tests. Both observed **89/90
lines** and **57/58 branches**. Their executed/missing/excluded line and branch
sets have the identical digest:

`4b83d9b05686c65b806ef165675c0ef375b59eafa909dd4c17bf775dcfd6cd50`.

Sysmon is isolated from testmon, coverage plugins, dynamic contexts and parallel
execution. This Python measurement makes no Rust coverage claim and changes no
coverage threshold or default core.

## Bounded mutation and positive control

Target: `scripts/check_migration.py`, SHA-256
`3e287565b831079a3de4048de72d3f17caf60b33930682b0acfd7f9ea5de0f49`.
Operators: `comparison,boolean`; 60-second coverage pre-scan, 10 seconds per
mutant, 180 seconds for the whole pytest process tree; serial, without mutation
cache reuse.

Initial trial: 64 ordinary tests, 34 mutants, 33 zapped and one survivor. The
survivor inverted the empty source-inventory guard. Adding a synthetic Git test
for empty, clean and dirty inventories killed that same mutation. Final trial:
**65 ordinary tests and 34/34 mutants zapped**, with zero survivors, errors,
timeouts, pardons or skipped files. No production guard or mutation operator was
changed to obtain this result. This does not qualify the native mutation gate or
other operator families.

The wrapper runs byte-identical staged policy/test/configuration inputs inside
its ignored run directory. Pinned Gremlins writes JSON relative to pytest's root,
and its subprocess import behavior requires the `scripts/__init__.py` marker.
Rejected isolation trials remain negative evidence: external test paths produced
34 mutant errors; omitting the package marker loaded the wrong source copy and
produced 34 survivors. Both exited zero. Final validation uses the report's
outcomes and unchanged input hashes, not process exit status alone.

## Review fixes and exclusions

Independent review reproduced an interrupted child continuing after the parent
released its lease and a failed warm run retaining its previous success stamp.
The runner now stops/reaps its process tree before releasing the lease and removes
the stamp before starting a database writer. Real signal/timeout tests and four
failed-run recovery cases pass. Skip-only runs cannot seed successful reuse.

VCR.py is excluded because the application HTTP client is native Rust `reqwest`,
already tested through synthetic local TCP fixtures. Freezegun does not control
native clocks and has no demonstrated active policy target. pytest-picked adds
no needed selection capability. No live HTTP recording was performed. Detailed
references and usage are in the
[maintainer guide](../../../docs/astro-site/src/content/docs/maintainers/test-acceleration.md).

## Remaining integration gates

Before publication, the branch was rebased onto
`13bdcf10ccb3e1cd06da1df4d98fc5da831768db`, which adds the independent ensemble
time-grid fix from PR #600. The acceleration implementation is unchanged by that
rebase. The 130 focused tests passed again in 8.27 seconds. An additional review
of the actual CI command found that pytest's console entrypoint did not add the
repository root to imports for two new test modules; their imports were made
explicit without changing CI's test command or scope.
The clean console-entrypoint command, with inherited `PYTHONPATH` removed, then
passed all 129 ordinary policy/profile cases in 5.25 seconds. The two changed
test files also passed Ruff lint and formatting.

Local locked native builds were attempted without changing pins. The release
build was killed with SIGKILL while compiling `argminmax` at two jobs. A dev build
with one job and debug information disabled was killed compiling `polars-core`.
Consequently, this record does **not** claim local full native, installed-consumer,
clean-package, or platform acceptance. Hosted required checks must qualify the
actual PR head. All 38 scientific migration obligations and separate release
readiness remain open. The track stays active until integration is qualified.
