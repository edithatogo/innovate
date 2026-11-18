# Integrated native qualification — 16 November 2026

## Result and source boundary

The integrated native source passes the bounded installed Arrow, native export,
provider/network and minimum-compiler checks below. This is fresh execution
after the external merge of #656 changed the workspace and native sources.
The earlier [122-input receipts](../arrow-installed-20261116/README.md) remain
unchanged and retain their original source scope.

The build started from local merge commit
`218c5a2f86037bf786d6292c7dbb3673a326d457`, tree
`81dffc475d082208cb5563747b685b4451cd6397`. Its parents are the published
reconciliation head `a49d453e8786eca78ad89055a442862b3afcf9d2` and main
`57d2916586d107c8affe90355e453abe27f01c26`.

[build-provenance.json](build-provenance.json) freezes **175 tracked build
inputs** before compilation. All 175 remained unchanged after the build and
each subsequent check. [source-comparison.json](source-comparison.json) records
53 added paths, seven changed paths and 115 unchanged paths relative to the
earlier manifest. The seven changed paths include both root Cargo files, the
core manifest, core library/OECD/regime sources and the tabular implementation.
Earlier native passes are not transferred to these changed inputs.

The Kairos revision remains
`a1f6283575ad26918230adc122a99b5c08fc4da3`. This packet writes only this evidence
directory; native code, existing tests, manifests, locks, provider pins, shared
CI and earlier evidence are unchanged by this qualification work.

## Fresh development wheel and installed consumer

The owned Rust 1.100.0 / Cargo 1.100.0 cache was reused with one build job and
the development profile. The locked build compiled the integrated core,
tabular and Python crates and produced a new wheel in a separate output
directory. The actual command and output are retained in
[build-provenance.json](build-provenance.json) and [build.log](build.log).

Artifact: `innovate-0.7.0a1-cp315-cp315-manylinux_2_35_x86_64.whl`.

SHA-256: `af3fad4f104f3b8ffad6ba92fe139f9c8a7183ee85e0a9ecd938f6e2dc015d3c`.

The wheel was installed into the owned Python 3.15.2 environment. Polars 2.1.0,
PyArrow 26.0.1 and pytest 9.2.0 match the declared pins; `uv pip check` passed.
[environment.json](environment.json), [install.log](install.log) and
[environment-check.log](environment-check.log) retain the selected package set
and actual commands. This checks those installed versions, without claiming
the complete development dependency closure or every artifact in `uv.lock`.

The unchanged qualifier executed with isolated Python imports outside the
checkout. It checked wheel RECORD entries and installed package bytes, the
compiled-extension loader, build/source/test identities and all five provider
pins before admitting the test result. Its SHA-256 remains
`fe894db45329c4fd9906aec6a77f8bd87653b0a7d927ac4aeaed39e6e099b3db`.

[qualification.json](qualification.json),
[qualification.junit.xml](qualification.junit.xml) and
[qualification.pytest.log](qualification.pytest.log) record **42 passed in
0.69 seconds, with zero skips, failures or errors**:

- All 38 existing scenario-export cases, including all four optional Arrow
  file/stream readback and round-trip cases.
- Four named existing Kairos/paused-scenario producer cases.

The qualifier and its guard tests were not modified. Their earlier guard and
coverage evidence remains separate from this fresh native execution; this
packet does not repeat or expand that tooling suite.

## Separate native Rust execution

Both commands used Rust 1.100.0, locked dependencies, one Cargo job and the
owned cache. Actual results and source/log hashes are retained separately.

| Target | Passed | Failed / ignored / filtered |
| --- | ---: | --- |
| `innovate-tabular`: `scenario_exports` | 8 | 0 / 0 / 0 |
| `innovate-core`: `kairos` | 17 | 0 / 0 / 0 |
| `innovate-core`: `network` | 37 | 0 / 0 / 0 |
| `innovate-core`: `network_diffusion` | 5 | 0 / 0 / 0 |
| `innovate-core`: `network_petgraph` | 5 | 0 / 0 / 0 |

The export command is recorded in [scenario-rust.json](scenario-rust.json) and
[scenario-rust.log](scenario-rust.log):

```sh
cargo test --locked -p innovate-tabular --test scenario_exports -j 1
```

The provider/network command passed **64 cases** and is recorded in
[provider-rust.json](provider-rust.json) and [provider-rust.log](provider-rust.log):

```sh
cargo test --locked -p innovate-core --test kairos --test network \
  --test network_diffusion --test network_petgraph -j 1
```

These native results are separate from the 42 installed Python cases. They
retain the named targets' existing expectations and make no claim that the new
scientific models or the complete native suite were executed.

## Actual minimum-compiler check

[msrv-check.json](msrv-check.json) and [msrv-check.log](msrv-check.log) record
successful execution on Rust **1.99.1** (`58cf8f9c3`) and Cargo 1.99.1:

```sh
cargo check --locked --workspace --all-targets -j 1
```

The command exited 0 on Linux x86_64 with default features, checking all five
workspace crates: `innovate-core`, `innovate-tabular`, `innovate-ffi`,
`innovate-python` and the newly included `innovate-kairos`. It reused the owned
minimum-compiler target cache and selected the owned Python 3.15.2 interpreter
for PyO3. The source manifest remained unchanged.

This establishes the checked compile configuration. It does not execute tests
on Rust 1.99.1, establish linked/installed release artifacts or qualify another
platform. `innovate-games` is explicitly excluded from the workspace; its files
are captured among the 175 source identities, without claiming this command
compiled that crate.

## Reproduction and remaining gates

Use the command and environment recorded in `build-provenance.json` to build a
new identified wheel, install that wheel into the selected owned environment,
and run:

```sh
python scripts/qualify_scenario_arrow.py \
  --repository /absolute/path/to/innovate \
  --wheel /absolute/path/to/identified.whl \
  --build-record /absolute/path/to/build-provenance.json \
  --output /absolute/path/to/new-qualification.json
```

The qualifier verifies the build record against current input and artifact
bytes and refuses to overwrite prior output. Executor build records are not
independent cryptographic attestations. The source identity here names the
observed local merge; publication and hosted receipts must retain their own
commit identities and establish any claimed source equivalence.

Recurring hosted Arrow/minimum-compiler execution remains a shared-CI intake.
Other-platform artifacts, provider publication/publishability, full scientific
qualification, security/supply-chain gates and integrator release acceptance
retain their existing requirements. **All 38 full scientific release
obligations remain open.**
