# Hosted evidence and integration intake — 16 November 2026

This receipt replaces the earlier pending observations **within the exact source
scopes below**. It does not transfer results to a later PR head. Source changes
from main are consumed unchanged in isolated merge
`2748084292e5e57c181c1665a686ad3a291145bf`, tree
`787eb4cd175b7343026cdad88143e8879e0b3062`.

[Job observations](job-observations.json) retain actual API states and step
outcomes. [Log extracts](job-log-extracts.json) retain exact log line numbers,
decoded full-response SHA-256 hashes and observed checkout/result lines. The
underlying public jobs are linked below. Skips and incomplete results are
reported separately from passes.

## PR #662 at d36c33f6

Head `d36c33f6b4fb480828ddb9aae0121f3379667015`; native matrix checkout
`cc48aaa0cfbad8830aff4a2feebcbec3cf4b2665`, GitHub's merge with then-main
`57d2916586d107c8affe90355e453abe27f01c26`. The previous receipt verified that
the synthetic merge and d36 head have the same tree. These are completed
historical results, not new-head admission.

| Lane | Observed result | Primary job |
| --- | --- | --- |
| Ubuntu native | Rust nextest 457 passed, 1 skipped; Python 487 passed, 5 skipped | [133267199616](https://github.com/edithatogo/innovate/actions/runs/43857612980/job/133267199616) |
| macOS native | Rust nextest 457 passed, 1 skipped; Python 487 passed, 5 skipped | [133267199772](https://github.com/edithatogo/innovate/actions/runs/43857612980/job/133267199772) |
| Windows native | Rust nextest 457 passed, 1 skipped; Python 485 passed, 7 skipped | [133267199946](https://github.com/edithatogo/innovate/actions/runs/43857612980/job/133267199946) |
| Python coverage, exact d36 checkout | 206/237 statements, displayed 87%; 487 passed, 5 skipped | [133265814316](https://github.com/edithatogo/innovate/actions/runs/43857613073/job/133265814316) |
| Rust coverage, exact d36 checkout | 21,046/24,089 lines, 87.37%; completed success | [133265814859](https://github.com/edithatogo/innovate/actions/runs/43857613073/job/133265814859) |

The Rust percentages above and below cover the workflow's explicit
`innovate-core`, `innovate-tabular` and `innovate-ffi` report, checked against
an 80% line threshold. They are not coverage of every workspace crate; no
branch-coverage percentage is inferred from the report's zero branch columns.

All nine jobs in the ordinary CI run completed successfully, including the
three Julia/R platform jobs, package, lint and aggregate jobs. Their status is
API-observed; this packet inspected full logs for the three native jobs and
both coverage jobs. The Python jobs used actual **3.15.2** despite the old
aggregate job label containing `3.15.1`.

The five Python skips on Ubuntu/macOS are one live OECD endpoint and four
optional Arrow cases. Windows adds two POSIX process-group checks. Ordinary
matrix success therefore does **not** execute the optional Arrow contract.
The separately built current-main wheel's [42-case installed packet](../../../../scientific_ecosystem_alignment_20261029/evidence/evidence-acquisition-20261116/README.md)
executes those four Arrow cases without skips.

## Current main b29a1d27

Main `b29a1d27f54fdd08d77881ee999772fceab7c03a` includes #668, merged at
12:13:59 UTC (22:13:59 Brisbane). It also includes the separate dependency and
Renovate changes in #667/#666/#670. No source/manifest pin is edited by this
scientific/journal continuation.

| Evidence | Observation | Primary job |
| --- | --- | --- |
| Strict native documentation | `cargo doc` with missing-docs/broken-link/invalid-codeblock denials succeeds; Rust doctests and maintained Python/Julia/R examples execute | [133290138712](https://github.com/edithatogo/innovate/actions/runs/43864696778/job/133290138712) |
| Whole documentation contract | Execution and current content pass; overall `complete:false` remains in the [actual contract report](main_native_docs-contract-report.json) | Same job |
| Browser qualification | **141 passed, 1 failed**: desktop navigation test reported an aborted `ui-core.D6xEXLGp.js` request; deployment skipped | [133290139096](https://github.com/edithatogo/innovate/actions/runs/43864696778/job/133290139096) |
| Installed Voiage consumer | Version 3.1.0 wheel installed alongside the built Innovate wheel; **19 passed**; qualification step and evidence upload succeeded | [133290137529](https://github.com/edithatogo/innovate/actions/runs/43864696535/job/133290137529) |
| Rust coverage | Exact b29 checkout; **21,289/24,089 lines, 88.38%**; completed success | [133290142620](https://github.com/edithatogo/innovate/actions/runs/43864696637/job/133290142620) |

Main's ordinary native/platform CI and both coverage jobs are completed success
in the retained API observation. The native documentation failure reported for
older PR #662 source has been repaired upstream; the browser failure above is
a different remaining failure. The Voiage job establishes its scoped installed
consumer execution, not every economic/scientific consumer or a release.

## Documentation owner and receiving order

Closed, unmerged [#659](https://github.com/edithatogo/innovate/pull/659) is
superseded by the same owner's draft [#671](https://github.com/edithatogo/innovate/pull/671).
The inspected active head is `070a04424dfb084758d1c0b201bf2980002ea81d` and its
base is b29. The final commit changes Conductor records only. The substantive
source result below belongs to `b2ed945f9ffede3bdd00c29437636c0bf1eba03d`,
checked out by GitHub at merge `96a02acc63a757bf6d07f679527c79bc1e1014a0`.
The complete later evidence tree is not represented as that earlier checkout.

- [Browser job 133296524820](https://github.com/edithatogo/innovate/actions/runs/43866570433/job/133296524820):
  **164 passed**. The added public-data/tutorial routes are included in the
  expanded suite. Deploy was correctly skipped for the PR.
- [Native documentation job 133296525320](https://github.com/edithatogo/innovate/actions/runs/43866570433/job/133296525320):
  strict Rustdoc and doctests pass; the [actual documentation report](docs671_native_docs-contract-report.json)
  records **complete:true**, 11 executable examples, 34 generated API pages,
  maintained language receipts and zero structural violations.
- Of #671's 520 changed paths, none overlaps this packet's two track namespaces.
  It does change **two assessed manuscript inputs**: the native migration guide
  and tabular `src/lib.rs`. The [disposable drift diagnostic](documentation-integration-drift.json)
  confirms that the current paper checker exits 1 on the changed guide.
  That expected failure is a handoff dependency, not a reason to auto-update hashes.

The sole integrator in #624 retains receipt, receiving order, shared-CI adoption
and promotion. Receive each owner's work, reassess the paper against the combined
source, rerun the render and installed native checks as applicable, and observe
required checks on the actual receiving head. The [complete Arrow/MSRV candidate](../../../../scientific_ecosystem_alignment_20261029/integration/README.md)
is prepared without modifying any active root workflow.

Neither documentation completeness nor hosted test success closes the 38
scientific migration obligations. The conditional offline-security runner is
still unqualified; the [current-main run observation](offline-run-observation.json)
records an actual queued run, not executed security evidence. No replacement runner or waived
check is introduced. Authorship, actual research use, selected release identity,
external registration and journal acceptance retain their own evidence.
