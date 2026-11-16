# Hosted qualification at b2ed945

PR [#671](https://github.com/edithatogo/innovate/pull/671) head
`b2ed945f9ffede3bdd00c29437636c0bf1eba03d` has passing
[documentation workflow 43866570433](https://github.com/edithatogo/innovate/actions/runs/43866570433)
and [native CI 43866570444](https://github.com/edithatogo/innovate/actions/runs/43866570444).
These receipts describe those runs, not a later evidence-only commit.

- [Browser qualification](browser-qualification.json): 164/164 cases (82 desktop,
  82 mobile); zero failures, skips, flaky or retried cases. Both mandatory
  homepage probes capture the original prefetched body and match the uploaded
  page, including each response once in the existing budgets. Managed Chromium
  is 159.0.8154.6; no executable override is used.
- [Runtime and platform qualification](runtime-qualification.json): installed
  Python, Julia and R documentation plus actual Windows/macOS/Linux CI results.
  Windows passes 422 Python tests with seven declared non-documentation skips;
  no catalogue or public-data case is skipped. Both earlier portability failures
  are closed by execution on Windows.
- Original producer receipts: [combined content execution](execution.json),
  [native API](native-api.json), [Julia](julia-documentation.json) and
  [R](r-documentation.json). The combined content gate passes with zero
  violations: 11 Python examples/117 assertion lines, 19 public doctests, seven
  Julia examples/ten assertions and six R examples/11 assertions.
- [Check metadata snapshot](check-snapshot.json) keeps the separate coverage,
  security and release states. Queued, running, skipped and neutral checks are
  not passes.

The browser receipt verifies all 215 tracked source inputs against the published
head and separately compares 35 derived API inputs. Its original browser
inventory contains 312 files; the Pages tar contains 311. The sole omission is
an empty `.nojekyll`, consistent with the pinned upload action's hidden-file
default. Every uploaded file matches the browser inventory. Both identities and
the packaging difference are preserved; no live deployment is inferred.

The source-only and locally incomplete Julia/R receipts remain unchanged under
`../integrated-5821d8a/`: later hosted execution closes the current implementation's
runtime gap without rewriting historical results. Full original artifacts and
logs were downloaded and their published ZIP digests verified; the bounded
receipts retain their identifiers, hashes and original locations.

Deployment was skipped. The existing 16-model/22-capability scientific release
hold, standalone games dependency conflict, offline runner qualification, wider
curriculum, human reader/assistive-technology evaluation and live publication
remain separate unfinished work. Automated browser acceptance does not complete
those obligations or establish independent build attestation.
