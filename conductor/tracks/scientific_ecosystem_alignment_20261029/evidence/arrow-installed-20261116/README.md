# Installed optional Arrow qualification — 16 November 2026

## Scope

ALIGN-02 executes the existing native scenario-export tests through an installed
wheel with the manifest-pinned optional PyArrow version. It also executes four
existing Kairos/paused-scenario producer cases. The new qualifier validates the
evidence boundary; it changes no scientific code, dependency, provider pin or
shared CI. Existing ALIGN-03 provider targets and the ALIGN-04 minimum-compiler
check are recorded separately below.

The worktree was at `0c718cea3eb982717ce940560868004d71261e9b` when the
development build began. This is a local reconciliation commit, not a claim that
the subsequently published PR head was compiled. The build record freezes all
122 tracked Rust/Python build-input hashes, including both locks, and records
uncommitted tooling. Qualification compares those bytes again after execution.
The Kairos pin is `a1f6283575ad26918230adc122a99b5c08fc4da3`.

## Execution and limitations

- Own Python 3.15.2 environment with Polars 2.1.0, PyArrow 26.0.1, pytest 9.2.0
  and maturin 1.16.0.
- Own copy of Rust 1.100.0, Cargo 1.100.0, Cargo cache and target directory.
  Other workers' environments and caches are not modified.
- One Cargo build job, development profile with repository `debug = 0`.
  Initial capacity: 50 GiB free disk, about 7.1 GiB available RAM, no swap.
- This is a bounded Linux development-wheel consumer check. It is not an
  optimized-release benchmark, Rust 1.99.1 minimum-version check, other-platform
  acceptance or closure of any of the 38 full scientific release obligations.

The first qualifier guard-test run preceded the script and failed because the
new module was absent. `qualifier-red.log` retains that result. It is test-first
tooling evidence, not a numerical or native-runtime failure. After implementation
and independent review, the 45 guard cases passed; they reject missing/drifting
dependencies, skipped, missing, duplicated or unexpected test cases,
installed/wheel/RECORD mismatches, unsafe member paths, changed test inputs and
incomplete/mismatched build records.
Ruff checks and formatting passed for the new script and guard tests.

## Executed installed result

The one-job Rust 1.100.0 development build completed successfully in 4 minutes
24 seconds. All 122 frozen build inputs were unchanged after compilation.
`uv pip check` accepted the installed package set. The compiled wheel is
`innovate-0.7.0a1-cp315-cp315-manylinux_2_35_x86_64.whl`, SHA-256
`0b9e5681c8c0803ae0f1bb583e9bf91f8c2b8182dfe3c86fc5e26c5d8474fc8d`.

[qualification.json](qualification.json) is the initial uninstrumented installed
receipt;
[qualification.junit.xml](qualification.junit.xml) and
[qualification.pytest.log](qualification.pytest.log) record **42 passed in
0.54 seconds, zero skips/failures/errors**. This includes all 38 scenario-export
cases, all four optional Arrow file/stream cases, and four Kairos/paused-producer
cases. The qualifier verified the wheel RECORD and all four installed package
members, the actual compiled-extension loader, the installed dependency versions,
isolated import paths and the test/fixture hashes before and after execution.

[build-provenance.json](build-provenance.json) and [build.log](build.log) retain
the actual source/toolchain/command and output identities. The receipt records
the qualifier's SHA-256, allowing readers to distinguish any later tool revision.
Independent read-only review confirmed that the 45 guard tests pass, the 42-case
selection is complete, and all artifact/source/test/log hashes match. No blocking
finding remained for this one-platform development-wheel scope.

The later [qualification-coverage.json](qualification-coverage.json) is the
latest isolated-worker receipt, collected to measure the new qualifier's code
coverage. It uses the same wheel and unchanged qualifier/source/test hashes and
again reports all **42 passed**, with zero skips/failures/errors. The original
uninstrumented receipt remains intact.

## Separate native and minimum-compiler evidence

[scenario-rust.json](scenario-rust.json) and [scenario-rust.log](scenario-rust.log)
record the named Rust export target:

```sh
cargo test --locked -p innovate-tabular --test scenario_exports -j 1
```

Rust 1.100.0 executed **8 passed, zero failed/ignored/filtered**. These tests
exercise native invalid-envelope/outcome rejection, success/failure/pause data
preservation and recovery. They are separate from the 42 installed Python cases.

[msrv-check.json](msrv-check.json) and [msrv-check.log](msrv-check.log) record a
separate owned Rust 1.99.1 (`58cf8f9c3`) / Cargo 1.99.1 execution:

```sh
cargo check --locked --workspace --all-targets -j 1
```

It exited 0 in **3 minutes 15 seconds**, checking the core, tabular, Python and
FFI crates on Linux x86_64 with their default features. All 122 build-input hashes
remained unchanged. This establishes the checked minimum-compiler configuration;
it does not execute tests, link/install a release artifact or qualify another
platform on that compiler. The declared minimum, dependency pins and locks were
not changed. Recurring hosted minimum-version enforcement remains a shared-CI
decision.

## Provider and network targets

[provider-rust.json](provider-rust.json) and [provider-rust.log](provider-rust.log)
record the current pinned Kairos provider's existing integration targets:

```sh
cargo test --locked -p innovate-core --test kairos --test network \
  --test network_diffusion --test network_petgraph -j 1
```

| Target | Passed | Failed / ignored / filtered |
| --- | ---: | --- |
| `kairos` | 17 | 0 / 0 / 0 |
| `network` | 37 | 0 / 0 / 0 |
| `network_diffusion` | 5 | 0 / 0 / 0 |
| `network_petgraph` | 5 | 0 / 0 / 0 |
| **Total** | **64** | **0 / 0 / 0** |

Cargo metadata confirms all four target names exist; the earlier `--test network`
command is valid. This is actual Rust 1.100.0 execution on the frozen native
inputs and Kairos pin, separate from both the eight Arrow export tests and the
installed Python cases. Upstream crate publication/publishability, other
platforms, full scientific validation and integrator acceptance remain open.

## Coverage and a real rejection control

[coverage-summary.json](coverage-summary.json) and
[qualifier-coverage.json](qualifier-coverage.json) combine the 45 guard tests,
the real isolated 42-case worker, and an actual malformed-build-record rejection
under coverage 7.17.3. The qualifier covers **196/215 statements (91.16%)** and
**84/100 branches (84%)**, with no excluded lines. The repository's 90% new-code
line target is met. This is coverage of the new evidence-checking script; no
Rust algorithm or whole-repository coverage is inferred.

The [rejection receipt](qualification-rejected-build-record.json) intentionally
contains `status: "failed"`: a synthetic build record containing JSON `[]`
produced exit 1 and the explicit error `build record must be an object`, before
any native tests ran. It checks failure-receipt emission for the issue raised by
independent review. It is not a failed scientific candidate or a passing native
result. Each execution used a new output path; the qualifier refuses to overwrite
existing evidence.

## Reproduction contract

Build and install the wheel in an owned environment outside the source checkout.
Use the exact runtime pins from `pyproject.toml` and a locked Cargo build with the
declared provider revision. The executor build record must contain:

- `schema_version: 1`, `status: "built"`, `build_exit_code: 0`;
- source commit/tree and dirty-path inventory at build time;
- `source_inputs`: every path/hash returned by `build_inputs(repository)` in
  `scripts/qualify_scenario_arrow.py`, captured before the build and rechecked
  after it;
- `wheel_sha256`, actual compiler/Cargo versions, build command, profile and job
  count, plus the retained build-log path and hash.

The executed environment used these exact declared versions and passed
`uv pip check`; this does not claim that the complete development-group closure
or every platform artifact in `uv.lock` was installed or tested.

Run the selected environment's Python:

```sh
python scripts/qualify_scenario_arrow.py \
  --repository /absolute/path/to/innovate \
  --wheel /absolute/path/to/identified.whl \
  --build-record /absolute/path/to/build-provenance.json \
  --output /absolute/path/to/qualification.json
```

The qualifier launches itself with `-I` outside the checkout, verifies every
wheel package member against its RECORD and installed bytes, and requires the
declared dependency versions. It copies only the two existing test files and
their checkpoint fixture into a temporary test tree, with hashes checked before
and after execution. Pytest runs in a further isolated process with plugin
autoload disabled and a fixed explicit selection: all 38 export cases, including
all four optional Arrow file/stream cases, and four native producer cases.

The command retains JUnit and output logs beside its JSON receipt. Every named
case must occur exactly once with no skip, failure or error. The record binds
these results to the tested artifact and source/test bytes. Build records remain
executor evidence; they are not independent cryptographic attestations.

Recurring hosted execution is still a shared-CI intake for the #624 integrator.
This packet supplies an executable gate and a bounded receipt, without editing
that owner's workflows or promoting a dependency or release.
