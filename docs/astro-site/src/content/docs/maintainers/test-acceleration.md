---
title: Local Test Acceleration
description: Bounded serial feedback profiles, cache validity, and evidence limits for the native migration.
---

The profiles in `scripts/test_acceleration/` provide optional local feedback for
[issue #473](https://github.com/edithatogo/innovate/issues/473). They support
dependency selection with testmon, a small Python mutation trial, and a comparison
of Coverage's measurement cores. Every receipt records `promotion_evidence: false`.
Full ordinary CI, native and installed-consumer tests, scientific obligations,
coverage thresholds, security checks, rights checks, and publication gates remain
unchanged.

These commands have been exercised locally on Linux/POSIX with Python **3.15.2**.
The runner contains Windows process-tree cleanup, but that path has not been
locally qualified; its process-cleanup regression tests exercise POSIX behavior.
The ordinary CI platform matrix does not establish support for these optional
accelerated profiles on every platform.

## Set up an owned environment

Run commands from the repository root. Select a repo-local environment you own,
and finish its active jobs before installing or rebuilding anything. The following
POSIX examples use `.venv-native`. Create it only for a new checkout or when that
environment does not already exist:

```sh
uv venv .venv-native --python 3.15.2
```

Select its executable explicitly, install the ordinary locked project, then add
the optional tools from their separate requirements file:

```sh
INNOVATE_PROFILE_ENV="$PWD/.venv-native"
INNOVATE_PROFILE_PYTHON="$INNOVATE_PROFILE_ENV/bin/python"
"$INNOVATE_PROFILE_PYTHON" -c 'import sys; assert sys.version_info[:3] == (3, 15, 2); print(sys.version, sys.prefix)'
CARGO_BUILD_JOBS=2 UV_PROJECT_ENVIRONMENT="$INNOVATE_PROFILE_ENV" \
  uv sync --locked --python "$INNOVATE_PROFILE_PYTHON"
uv pip install --python "$INNOVATE_PROFILE_PYTHON" --require-hashes \
  -r scripts/test_acceleration/requirements.txt
```

The optional requirements pin `pytest==9.2.0`, `coverage==7.17.2`,
`pytest-testmon==2.3.0`, and `pytest-gremlins==1.13.2`, with hashes for their resolved
dependencies. The runner checks the versions required by each profile. The
production `pyproject.toml` and `uv.lock` are unchanged. A later exact `uv sync`
can remove optional packages; reinstall the requirements afterward. Invoke the
selected Python directly for profiles so an implicit environment synchronization
does not change the experiment.

To qualify changed-code selection and recovery against a synthetic repository,
run the opt-in integration test after installing these tools:

```sh
"$INNOVATE_PROFILE_PYTHON" -m pytest tests/test_acceleration/test_testmon_integration.py
```

This test verifies affected-test selection, failure recovery, and full invalidation
after native, fixture, manifest or inventory changes. It is outside the default
CI test path and requires the optional tools; missing tools produce a failure.

### Rebuild native code before native feedback

The `native` scope requires the installed `innovate` extension to correspond to
the source under test. After a Rust, Cargo, build configuration, or binding change,
rebuild and reinstall through the ordinary project workflow before using cached
selection. An explicit project reinstall is:

```sh
CARGO_BUILD_JOBS=2 UV_PROJECT_ENVIRONMENT="$INNOVATE_PROFILE_ENV" \
  uv sync --locked --python "$INNOVATE_PROFILE_PYTHON" --reinstall-package innovate
uv pip install --python "$INNOVATE_PROFILE_PYTHON" --require-hashes \
  -r scripts/test_acceleration/requirements.txt
```

The runner fingerprints native source inputs and any installed `_native` extension
it discovers. Those hashes identify the bytes observed. They do **not** prove
which source, compiler, or build configuration produced a binary, and invalidating
testmon does not rebuild Rust. Preserve normal build and installed-consumer
evidence. The `policy` scope can exercise the Python policy checks without a
native extension.

## Choose a profile

All profiles run serially. There is no passthrough for arbitrary pytest selectors,
workers, retries, or plugin options.

| Profile | Scope | Purpose |
| --- | --- | --- |
| `baseline` | `native` by default, or `policy` | Execute the complete chosen scope with measurement hooks and no selection plugin. |
| `testmon` | `native` by default, or `policy` | Seed dependencies or run the tests selected from a valid local database. |
| `coverage-ctrace` | `policy` only | Measure the policy target's lines and branches with the C tracing core. |
| `coverage-sysmon` | `policy` only | Measure the same target with `sys.monitoring`, verifying the actual core. |
| `gremlins` | `policy` only | Mutate one Python policy file with two operator families. |

The `native` scope selects `tests/native`. The `policy` scope selects
`tests/native/test_migration_policy.py` and
`tests/native/test_migration_mutation.py`. Gremlins deliberately uses only the
second file, whose ordinary tests import the policy through Python's normal
import machinery so the plugin can instrument it.

Start with a comparable baseline and an explicit seed:

```sh
"$INNOVATE_PROFILE_PYTHON" -m scripts.test_acceleration baseline --scope native
"$INNOVATE_PROFILE_PYTHON" -m scripts.test_acceleration testmon --scope native --rebuild
"$INNOVATE_PROFILE_PYTHON" -m scripts.test_acceleration testmon --scope native
```

Use the smaller policy scope when working on migration policy:

```sh
"$INNOVATE_PROFILE_PYTHON" -m scripts.test_acceleration baseline --scope policy
"$INNOVATE_PROFILE_PYTHON" -m scripts.test_acceleration testmon --scope policy --rebuild
"$INNOVATE_PROFILE_PYTHON" -m scripts.test_acceleration testmon --scope policy
```

`--rebuild` is valid only for testmon. It discards that scope's dependency database
and runs its whole scope with `--testmon-noselect`. It does not clear another
scope's database or repair a native build.

The default `--timeout 180` bounds the main pytest process tree; accepted values
are 1–300 seconds. A timeout produces a failed result, with process exit code
`124` in the receipt. Snapshotting, cleanup, and report generation add overhead;
this option is not a strict deadline for every operation in the wrapper. Coverage
report generation has a separate subprocess budget using the same value.

## Keep testmon's cache valid

[Testmon records dependencies using Coverage.py](https://www.testmon.org/).
Python execution tracing cannot determine all effects of native code, data files,
subprocesses, or external state. This runner therefore uses a deliberately broad
invalidation policy:

| Change | Next testmon run |
| --- | --- |
| Python content under `bindings/python/`, with the file inventory unchanged | Eligible for testmon's dependency selection. |
| Test contents, parametrization, fixtures in test modules, or added/removed files | Discard the database and execute the full scope. |
| Other tracked or nonignored inputs, including Rust, lockfiles, configuration, scripts, data, documentation, or evidence | Discard the database and execute the full scope. |
| Interpreter identity, platform, installed dependency versions, or scope | Discard the database and execute the full scope. |
| Discovered installed native extension bytes | Discard the database and execute the full scope. |
| Missing database, missing successful state, or `--rebuild` | Execute a fresh full seed. |

The snapshot includes tracked files and nonignored untracked files. Symlinked
source inputs are refused. Ignored output directories are excluded, so receipts
do not invalidate themselves. Changes during a run produce
`inputs_changed_during_run`, preventing reuse of that result as a successful seed.

This does not capture every environment variable, ignored input, external file,
or service state. If such an input affects a test, stabilize it and explicitly
rebuild the cache. Keep the ordinary suite as the acceptance check. Unexpected
selection-sensitive failures should be investigated as possible
[hidden test dependencies](https://www.testmon.org/blog/hidden-test-dependencies/).

Each scope owns its database under
`.pytest_cache/test_acceleration/testmon-<scope>/`. A single atomic
`.pytest_cache/test_acceleration/active.lock` directory prevents concurrent profile
writers across scopes. The runner refuses a second writer. After an abnormal
termination, confirm that the owning runner and its subprocesses have stopped
before removing a stale empty lease directory, then run `testmon --rebuild` for
the affected scope. Never remove another active process's lease.

Plugin autoloading is disabled within profile subprocesses; each profile loads
its own measurement and instrumentation plugins explicitly. The runner refuses
CI execution and incompatible inherited settings, including pytest selectors or
plugins, xdist worker state, Coverage settings, testmon database overrides, and
the live OECD opt-in. Unset the named setting in the local shell before retrying.
Keep testmon, pytest-cov, standalone coverage, mutation testing, and parallel
execution in separate runs.

## Interpret receipts and failures

Each run prints a JSON receipt and retains its receipt, counts, and log in a
unique directory under `.pytest_cache/test_acceleration/`. `local_receipt` points
to that directory's `receipt.json`. Raw local logs and mutation reports can
contain test identities, failure details, and local paths. Review and sanitize
them before sharing; retain only appropriate evidence in the issue's track.

| Field | Meaning |
| --- | --- |
| `pytest.collected` | Items observed in this invocation: selected plus deselected items. Testmon can avoid collecting unchanged files entirely. |
| `pytest.selected`, `pytest.deselected` | Actual item counts reported by this invocation's pytest hooks. |
| `pytest.started`, `pytest.executed` | Unique tests that started and unique tests that reached their call phase. Setup skips need not reach a call. |
| `pytest.passed`, `pytest.failed`, `pytest.skipped`, `pytest.errors` | Actual outcomes; collection and setup/teardown errors are separate from failed test calls. |
| `seed_inventory_count` | Historical inventory from a successful full testmon seed. It is not the current collected or executed count. |
| `pytest.duration_seconds` | The pytest session interval measured by the metrics plugin. |
| `pytest_wall_seconds` | Time awaiting the main pytest subprocess, including plugin startup and any mutation work it performs. |
| `wall_seconds` | Wrapper elapsed time, including snapshots, subprocess work, and result validation; excludes environment setup, native builds, and interpreter startup before entering the runner. |
| `snapshot`, `tools`, `invalidation_reasons` | Input fingerprints, relevant pinned versions, and why a full dependency seed was necessary. |

`passed` requires fresh passing tests and no reported failures or errors. A valid
warm testmon run with a nonempty successful seed may instead return
`no_tests_selected`, including when pytest exits with code 5. The wrapper exits
successfully for that state, but it records **no new verification**. Never report
the cached seed inventory as tests executed in an entirely deselected warm run.

Empty or all-skipped cold runs, failed tests, collection errors, missing metrics,
timeouts, and changed inputs cannot create a reusable success stamp. The state
stamp is removed before testmon starts, so an interruption cannot preserve an old
success marker beside a partially updated database. Some refusals or interruptions
occur before a complete receipt can be written. The wrapper returns nonzero for
failed or incomplete outcomes; it does not retry them automatically.

For performance decisions, compare ordinary baseline, cold seed, warm unchanged,
and changed-source runs on the same scope, interpreter, dependency set, and native
build. Compare both subprocess and wrapper time, and retain actual selected and
executed counts. The [issue #473 track](https://github.com/edithatogo/innovate/tree/main/conductor/tracks/test_acceleration_20261114)
is the location for measured receipts, source identities, and qualification
limits. A warm run that executes nothing is not evidence that equivalent testing
became faster.

## Compare Coverage's cores

```sh
"$INNOVATE_PROFILE_PYTHON" -m scripts.test_acceleration coverage-ctrace --scope policy
"$INNOVATE_PROFILE_PYTHON" -m scripts.test_acceleration coverage-sysmon --scope policy
```

Both profiles measure `scripts/check_migration.py` with branch coverage enabled,
the same policy tests, and an isolated configuration. The runner requests
`--debug=core` and checks that the observed core in `core.log` exactly matches the
requested `ctrace` or `sysmon`. A silent fallback cannot qualify the profile.

Compare the two receipts' `coverage.measurement_sha256` on unchanged inputs. That
digest covers executed, missing, and excluded lines plus executed and missing
branches. Also review the coverage summaries and raw `coverage.json` reports.
Passing each command independently verifies its own report; equivalence between
the two measurements must still be checked before interpreting timing differences.

[Coverage 7.17.2's core documentation](https://coverage.readthedocs.io/en/7.17.2/config.html#run-core)
describes sysmon's restrictions around plugins, dynamic contexts, and some
concurrency libraries. The profile uses the supported Python 3.15.2 branch
measurement path, with no dynamic contexts or additional coverage plugin. Testmon
has its own tracing requirements and is not run under this profile. Neither core
measures Rust execution or changes any coverage acceptance threshold.

## Run the bounded mutation trial

```sh
"$INNOVATE_PROFILE_PYTHON" -m scripts.test_acceleration gremlins --scope policy
```

The pinned Gremlins profile targets only `scripts/check_migration.py`, with
`comparison,boolean` operators, a 60-second coverage pre-scan limit, a 10-second
per-mutant limit, and the wrapper's default 180-second pytest process-tree budget.
It uses serial execution with mutation-result caching disabled. Gremlins 1.13.2
does not provide a maximum-mutant-count option; the single file, operator choice,
and time budgets bound this trial. See the
[pinned configuration reference](https://github.com/mikelane/pytest-gremlins/blob/v1.13.2/docs/guide/configuration.md)
and [report implementation](https://github.com/mikelane/pytest-gremlins/blob/v1.13.2/src/pytest_gremlins/plugin.py).

This plugin writes JSON below pytest's root directory. To keep output in the
ignored cache and ensure subprocesses import the instrumented file, the runner
stages exact copies of five inputs in its run directory:

- `scripts/__init__.py`
- `scripts/check_migration.py`
- `tests/native/test_migration_mutation.py`
- `migration-ledger.json`
- `pyproject.toml`

It also copies the metrics plugin to `_innovate_test_metrics.py`. The staged root
is the working directory, pytest root, configuration root, and Python import
path. The five input hashes appear in `mutation.staged_inputs_sha256`, and the
runner checks that those staged bytes remain unchanged. The package marker and
configuration are necessary: importing a different copy of the policy can make
mutations appear to survive without ever activating them.

The initial qualified trial used 65 ordinary policy tests and generated **34
mutants**, all zapped, with zero survivors, timeouts, errors, or pardons. This
result covers those operator families in that Python policy file. It establishes
no mutation score for Rust, numerical algorithms, other Python files, or omitted
operator families. A source-inventory guard initially survived; adding an
ordinary regression for empty, clean, and dirty source inventories caused that
same guard mutation to be killed. This supplied a concrete sensitivity check.

The wrapper reads `coverage/gremlins/gremlins.json` inside the run directory.
Gremlins' process exit code alone cannot qualify a mutation run: baseline tests
can pass while mutants survive or error. Missing, empty, skipped-file, or
inconsistent reports fail validation. Any survivor, timeout, error, or pardon
prevents `all_killed`, produces `mutation_incomplete`, and makes the wrapper exit
nonzero. Preserve those outcomes for investigation. The new policy tests are
also discovered by ordinary `pytest tests/native`, so their regression coverage
participates in the existing complete suite.

## Why the other candidate tools are excluded

| Candidate | Decision for the current code boundary |
| --- | --- |
| VCR.py | Its [supported clients](https://vcrpy.readthedocs.io/en/stable/installation.html#compatibility) are Python HTTP libraries. OECD retrieval is implemented with Rust `reqwest` in `rust/crates/innovate-core/src/oecd.rs`, with synthetic local TCP fixtures. Python interception would not cover that transport. No live HTTP cassettes or restricted payloads are recorded. |
| Freezegun | Its [documented behavior](https://github.com/spulec/freezegun#usage) patches Python date/time APIs, including monotonic and performance clocks. The scientific inputs use explicit time values; Rust transport deadlines use Rust clocks. There is no demonstrated wall-clock-sensitive policy test that benefits from freezing time, and real timeout behavior must remain measurable. |
| pytest-picked | Its [documented selection](https://github.com/anapaulagomes/pytest-picked#features) uses modified test files and Git state. That cannot establish dependencies from changed Rust, fixtures, configuration, or unchanged tests that exercise changed production code. Adding another selector would not resolve the native dependency boundary. |

Reconsider a tool only when a concrete test exposes a relevant gap, with a
bounded fixture and measured benefit. The current profiles provide local
feedback while the ordinary complete gates continue to determine acceptance.
