# PR 610 evidence-upload repair

## Scope and source identity

Base PR head: `3a8c8f6873c69e3e1f5be8f17857e10fb5b4c7c8`.
Receiving main: `30fa17d5fa5e40e05cd4313032de4921a3971f52` (the PR head's parent).
Inspected workflow blob: `a6933aec6d51ac661519ef8734fe90f0b1f38dd9`.

This bounded integration repair changes only `.github/workflows/docs.yml`,
`tests/test_documentation_workflow.py`, the owning `plan.md` and this evidence file.
It does not change native implementations, dependency/toolchain pins, scientific
migration obligations, branch protection or the Pages activation condition.

## Observed failure

Documentation run `43637664744`, build job `132577948093`, completed its six
Python contract suites, Node tests, Astro build/type check, structural ratchet
and production-artifact verification successfully. It then failed at
`Retain documentation evidence`: no files found under
`docs/astro-site/.reports/`. The upload action logged
`include-hidden-files: false`. Consequently, the native-documentation-examples
job was skipped, not qualified.

The pinned upload-artifact action excludes files in hidden directories unless
explicitly enabled. Both evidence upload steps had the same configuration bug.
The repair opts in for both steps, but uploads only four explicitly named JSON
receipts per job. It does not upload the entire hidden reports directory, use a
wildcard, or expose unrelated hidden workspace files. `if: always()` preserves
failure evidence; `if-no-files-found: error` continues to fail for empty output.
The validators, not artifact upload, remain responsible for receipt completeness.

## Executed bounded local verification

The connected workstation was unavailable and this sandbox could not clone the
repository. The exact workflow was materialized into an isolated partial working
tree and its Git blob SHA independently matched the value above. The sandbox
interpreter was **CPython 3.13.5**, not the required hosted **3.15.2**; these results
are supplementary configuration regressions, not complete repository acceptance.

- RED: six dependency-free unittest cases ran against the original workflow;
  three failed: both hidden receipt upload contracts and absent suite wiring.
- GREEN: the same six cases passed after the repair. Negative controls reject
  disabled/missing hidden-file inclusion, warning-only empty uploads, wildcards,
  traversal paths and unrelated private/hidden files.
- Independent YAML parsing and before/after semantic comparison confirmed that
  only the two upload input blocks and new regression-suite invocation changed.
  Deployment dependencies, activation, permissions, action pins, concurrency,
  timeouts, native commands and existing test suites are unchanged.
- `git diff --check` and Python bytecode compilation passed.

The regression suite is added to the existing documentation-contract loop so
hosted CI runs it on the required interpreter before building the site. Actual
artifact upload and all native examples still require a successful fresh hosted
run on the repaired head. No skipped, historical or queued job is counted as a
pass. No new complete native, Julia/R, browser, scientific or publication
qualification is claimed by this repair. Original local timing/source and
qualification records are retained unchanged.
