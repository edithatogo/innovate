# Independent hosted and native evidence review

Reviewed on 16 November 2026 by the independent scientific-evidence lane.
**No unresolved material discrepancy was found in the stated source scopes.**
This was a read-only comparison of retained observations, full job responses,
receipts, artifacts and Git objects. It did not rerun jobs or scientific tests,
publish an integration, or admit a new release.

## Review inputs and integrity

The [hosted account](README.md), [job observations](job-observations.json),
[log extracts](job-log-extracts.json), both documentation-contract reports and
the [drift diagnostic](documentation-integration-drift.json) were checked
against the retained full decoded job-log responses. All **11 response hashes**,
line counts and **249 selected lines** match those responses. Both complete
documentation reports equal the JSON objects parsed directly from their
respective full logs.

The [current-main native packet](../../../../scientific_ecosystem_alignment_20261029/evidence/evidence-acquisition-20261116/README.md)
was checked against its actual build, installed-test and compiler receipts.
All **178 recorded build inputs** match all four comparison points: published
main `b29a1d27f54fdd08d77881ee999772fceab7c03a`, the local build commit,
current source files and the installed qualification's input inventory. Its
qualifier and test-source hashes also match the current files.

The retained complete commit object hashes to
`2748084292e5e57c181c1665a686ad3a291145bf` and equals the actual local Git object.
This validates the local build identity; it does not pretend that the whole
repository tree equals public main or that a later publication commit has the
same ID.

## Checkout and result interpretation

| Evidence | Independently confirmed scope |
| --- | --- |
| PR #662 native matrix | Actual checkout `cc48aaa0…` merges head `d36c33f6…` with then-main `57d29165…`. Its tree and the head's tree both equal `b20fa0aa8c134db25aad27f04f762e400099bd02`. Each platform reports 457 Rust passes and one skip. Ubuntu/macOS report 487 Python passes and five skips; Windows reports 485 passes and seven skips. |
| PR #662 coverage | Both logs check out exact `d36c33f6…`. Python covers 206/237 statements, displayed as 87%; Rust covers 21,046/24,089 lines, displayed as 87.37%. |
| Main coverage | Exact `b29a1d27…` checkout covers 21,289/24,089 Rust lines, displayed as 88.38%. |
| Main documentation | Strict Rustdoc, doctests and maintained language examples pass. The actual contract still reports `complete:false`; its current content/execution pass does not remove the incomplete historical/full scope. Browser qualification reports 141 passes and one aborted-resource failure; deployment is skipped. |
| Documentation PR substantive source | `b2ed945f…` and checkout `96a02acc…` share tree `f826196927cc994410f9391b4381929d3641bf1a`. Its browser job reports 164 passes; its documentation report has `complete:true`, 11 examples, 34 generated API pages and zero structural violations. |
| Documentation PR later head | `070a0442…` has a different tree, `7725e89729437ce7e75a64a83a65dcbf6471ad8e`. Its delta from substantive source is confined to Conductor records. The account correctly does not transfer whole-tree hosted acceptance to this later head. |
| Main Voiage consumer | The log installs the 3.1.0 wheel alongside Innovate, reports 19 passes, and records successful qualification/evidence upload. This remains the named consumer scope. |

Skip accounting matches the full logs: Ubuntu/macOS omit one live OECD case
and four optional Arrow cases; Windows additionally omits two POSIX process
checks. The installed Arrow packet executes the four optional cases separately.
All nine ordinary PR #662 CI jobs and both coverage jobs are recorded as
completed success. The retained main CI/coverage observations likewise report
success, while preserving the separate browser failure.

Coverage arithmetic is correct. Both Rust line reports explicitly select
`innovate-core`, `innovate-tabular` and `innovate-ffi`, with an 80% line threshold.
They are not all-workspace coverage; the zero branch columns provide no measured
branch-coverage percentage. The Python statement percentage is rounded from
86.9198%, consistent with the displayed 87%.

## Native artifact and minimum compiler

The newly built wheel's SHA-256 matches its build and installed receipts. Its
eight hashed `RECORD` entries match the archived member bytes and sizes; all
four installed package members match their corresponding wheel entries. The
installed native extension's hash matches the qualification record. The copied
qualification JSON is byte-identical to its original output.

The build log, build record, pytest log, JUnit and MSRV log hashes all match.
The JUnit contains exactly **42 cases**, with zero failure, error or skip
elements: 38 scenario-export cases, including four optional Arrow cases, plus
four native producer/replay cases. Runtime versions are Python 3.15.2,
Polars 2.1.0, PyArrow 26.0.1 and pytest 9.2.0. The artifact remains a local
Linux development wheel, not an sdist-derived release.

The recorded Rust/Cargo 1.99.1 command is
`cargo check --locked --workspace --all-targets -j 1`. Its log checks all five
workspace crates and completes successfully. The account correctly limits
this to Linux/default-feature compilation; excluded games, test execution,
linking, other platforms and release acceptance are not inferred.

## Integration diagnostic and resolved review points

The Git comparison confirms **520 changed paths** in #671, no overlap with
this acquisition's two owning track namespaces, and exactly two changed
manuscript-assessed inputs: the native migration guide and tabular `src/lib.rs`.
The preserved disposable diagnostic reports the expected input-digest failure
on the guide. This review checked the affected-path mapping and recorded
failure, without rerunning that diagnostic. Reassessment on the actual combined
source remains a receiving dependency.

Two requested clarifications were applied and inspected:

1. The hosted README now names the three-crate coverage scope, the 80% line
   gate and the absence of a branch-coverage inference.
2. The previously unlinked security-queue claim now has a
   [dated run observation](offline-run-observation.json) for main run
   `43864697154`, recorded queued at 13:40:47 UTC. It is described as pending
   execution and does not qualify a runner or its assets. This review checked
   the retained API observation; it did not independently requery the provider.

All relative links in the reviewed hosted and native READMEs resolve. Existing
historical receipts remain separate. The 38 scientific obligations, actual
research use, author declarations, selected release identity, external
registration and journal acceptance retain their own evidence requirements.
