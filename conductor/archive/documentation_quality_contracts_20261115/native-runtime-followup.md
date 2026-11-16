# Native documentation runtime follow-up

## Evidence scope

This note records checks observed on 2026-11-14 UTC, before the workspace
executor became unavailable. The hosted language receipts below belong to main
`f6c7d058466ae5fe15b9846769cc0e6ba6774847`. The independent Python checks belong to
PR #611 head `c8a0fb6986537c424a2c47d7a44d15f746723d7a`.
They do not qualify later source, the new Julia runner edit, or the browser suite.
PR #611 subsequently merged as `384ed3b0f42d967f520dc66ef536d8fa0d92df7d`.

## Independent PR #611 verification

The live head was checked before creating the detached qualification worktree.
The actual repository Python 3.15.2 environment executed all seven complete test
modules, including the new nine-case source-hash contract module. Result:
**207 passed, zero failures, errors or skips, in 6.387 seconds**. This used the
complete modules, not extracted checker functions or a substitute interpreter.
The detached worktree remained clean.

| Module | Passed tests |
| --- | ---: |
| `tests/test_api_documentation_contract.py` | 29 |
| `tests/test_native_documentation_contract.py` | 28 |
| `tests/test_documentation_content_contract.py` | 60 |
| `tests/test_production_docs_contract.py` | 21 |
| `tests/test_api_reference_generation.py` | 22 |
| `tests/test_polyglot_api_documentation.py` | 38 |
| `tests/test_documentation_source_hash_contract.py` | 9 |

## Actual hosted Julia and R execution

[Main CI run 43640667982](https://github.com/edithatogo/innovate/actions/runs/43640667982)
passed. The three platform jobs each executed Julia **1.13.4** and R **4.6.2**.
Every platform reported **24 Julia docstrings, seven executed Julia examples and
nine observed assertions**, plus **24 R exports, 14 manuals, six executed R
examples and ten observed assertions**.

| Platform | CI job | Receipt-recorded native-library SHA-256 |
| --- | --- | --- |
| Linux | `132587318162` | `ef834c500ba3a27847ddabcd99e3565a2cae87cc74c642ab00b92bce7305feae` |
| Windows | `132587318219` | `2582d62640c8dc8e523dfc87a8cc18601eab87641951bb659a01eb54fad1dd22` |
| macOS | `132587318249` | `db9bdba99f185c465b51ea46412efa5fcead5a3adafa4d2680d2bafb325b4ed0` |

Julia and R recorded the same native-library hash within each platform.
The library bytes themselves were not independently downloaded and rehashed in
this follow-up; these are the actual hosted receipts' recorded identities.

All receipts reference inventory SHA-256
`694a437e0785f6036d6861f853538ebf02f050fa56f71f0aaf096f05e451ba59`.
All 23 recorded source fingerprints were independently checked against the
specified main revision with the inventory's newline normalization. Each example
ID, source path, source line, code digest and observed assertion count matched.

The downloaded archive bytes were independently hashed against their published
artifact digests before their JSON receipts were inspected:

| Artifact | ID | Verified archive SHA-256 |
| --- | --- | --- |
| Linux binding conformance | `14131823241` | `e34794e197f1016abb3da83ff07532ed2b098238e93c0d0f1ecb409f87084981` |
| Windows binding conformance | `14131944211` | `2c3f1f069348d45e61b3b541948fc9f24234998456e9b3532eadc53d29c43749` |
| macOS binding conformance | `14132283091` | `ee51ceb5a8cc17f022cf9bacb2a4aa29da737c9dec5bb7891001b05ded2207fb` |
| Main documentation execution | `14132707231` | `f05353f49481a53ea8681f7e8b89d47ecba20661fadd838fe4fa91b08f0edd1b` |

## Aggregate failure and its existing owner

Final PR #610 head `8a19659783f9e9490dbeb965d83d9090005644f8` passed CI run
`43639364211`, while documentation run `43639364235` failed aggregation after
individual runners passed. On main,
[documentation run 43640668020](https://github.com/edithatogo/innovate/actions/runs/43640668020)
built the site successfully; Native Documentation Examples job `132587656328`
failed at the aggregate contract and deployment was skipped.

The aggregate rejected both language receipts with
`Invalid polyglot source path or hash.` The inventory stores source hashes as
`sha256:<hex>`, whereas the aggregate validator expected bare hexadecimal.
PR #611 already owns the narrow repair: remove the source-hash prefix only when
validating and comparing that source digest, while retaining exact receipt/source
maps and the separate strict native-binary hash format. This follow-up does not
duplicate or broaden that repair. The independent 207-test result includes its
nine new regression cases.

These results do not waive the separate outstanding scientific release gates.

## Julia world-age warning repair

The actual hosted Julia logs also warned once per executed example about reading
`__documentation_assertions__` in a world older than the binding's definition.
The warning explicitly recommends `julia --depwarn=error` to turn the warning into
an error. The documented command-line flag has that meaning; using it for the
new hosted qualification is appropriate.

The runner now constructs `assertion_counter = Ref(0)` in the calling function,
interpolates that same object into the sandbox's constant binding, and reads the
retained local reference after `include_string`. It no longer retrieves a newly
introduced global binding from the older calling world. The `GlobalRef` used by
the assertion instrumenter, post-success increments, planned assertion count,
and observed-count equality checks are unchanged. This preserves detection of
skipped, repeated or failed assertions.

The rationale follows Julia's primary documentation:

- [World age](https://docs.julialang.org/en/v1/manual/worldage/): global bindings
  participate in world age, and a running function's world remains fixed.
- [Metaprogramming](https://docs.julialang.org/en/v1/manual/metaprogramming/):
  expression interpolation can embed an existing runtime object.
- [Command-line interface](https://docs.julialang.org/en/v1/manual/command-line-interface/):
  `--depwarn=error` converts deprecation warnings into errors.

No local Julia runtime was available, so the new runner source has **not yet
executed**. The meaningful regression is a fresh hosted run under
`--depwarn=error`, retaining seven Julia examples and nine observed assertions.
A textual Python imitation would not establish Julia runtime behavior.

The integrator must regenerate the polyglot inventory after the runner and CI
workflow edits, then regenerate the API reference and obtain fresh receipts.
Changing ordinary site prose alone does not require native-inventory regeneration;
changing this runner, native binding sources, R manuals, the inventory checker or
its recorded CI input does. Historical passing receipts must not be copied into
current reports as if they qualified the new hashes.
