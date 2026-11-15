# Track 5 completed candidate and ownership handoff

## Disposition

**Ownership and acceptance hold.** This packet preserves implementation and
independent numerical evidence completed during the bounded assignment. A later
live ownership check found the overlapping [PR #625](https://github.com/edithatogo/innovate/pull/625)
and the coordinator's instruction against a second statistics writer. Further
production edits, rebasing, conflict resolution and active-track rewrites stopped.

The existing draft [PR #622](https://github.com/edithatogo/innovate/pull/622)
retains its published claim and pre-migration characterization checkpoint.
This handoff update adds evidence files only. The completed migration is saved
as [candidate-implementation.patch](candidate-implementation.patch), **deferred
and not applied to the PR's production files**. No competing migration is
promoted, and neither packet is closed, deleted, archived or merged here.

Sole integrator: `6aeeeda6-d4ec-8320-a3b5-c3e2362e5b53`. The integrator must
reconcile both packets before selecting the accepted source and shared files.
This is implemented, tested code preserved for that decision; it is not a
replacement plan or a claim that Track 5 is complete.

## Immutable ownership anchors

| Record | Revision and observed status |
| --- | --- |
| Initially inspected main | `42189a223fd44c823ca11aec689ea67b11c25790` |
| #622 scope claim | `94c47a60ffa9db55f803e64b4823f2e9a7dc7c20`; PR opened at 06:31:50 UTC |
| #622 independent baseline | `b344f379926e934be30edc0842779dc5e3455f86` |
| Main after externally merged #621 | `1e190cfb29be95e0622e79c06bdcb4e272bea1d6`; #621 merged at 06:38:32 UTC |
| Overlapping #625 | `a39f81ca318c6ec98ac6ea3a62f9c99448dd0a21`; PR opened at 06:40:07 UTC |
| Coordinator map #626 | `4c88f396f22c2b6fa641f21460c5444a5633edd8`; draft ownership receipt |
| Later observed #625 head | `d949014164899d22f2fa2265bfa6f54e7dd6c939`; a main-merge commit recorded at 07:00:16 UTC, not tested by this packet |

The initial live inventory showed no competing statistics PR or branch claim.
That was a time-bounded observation. Although #622's PR opened first, #625
contains implementation commits recorded at 06:12:24 and 06:23:02 UTC. Those
times do not establish when the work became discoverable or authorize either
lane to displace the other.

[Issue #624](https://github.com/edithatogo/innovate/issues/624) reserves shared
integration, and [the map in #626](https://github.com/edithatogo/innovate/blob/4c88f396f22c2b6fa641f21460c5444a5633edd8/conductor/dependency-migration-ownership-20261115.md)
recognizes #625 without acknowledging #622. The discrepancy is handed to that
coordinator. The current main and #625 are not qualified by this packet's
original-base test runs.

### Overlap at the first conflicting head, `a39f81ca`

| Existing path | #625 proposal |
| --- | --- |
| `rust/crates/innovate-core/src/bayesian.rs` | Quantile, Gaussian scores and ESS changes |
| Active track `index.md` | Rename to archive |
| Active track `metadata.json` | Rename to archive |
| Active track `plan.md` | Remove active file and add archived plan |
| Active track `spec.md` | Remove active file and add archived specification |

#625 additionally proposes shared core manifest, lock, common registry and
scientific-review fingerprint changes. Its absence of this packet's dedicated
fixtures/tests does not independently clear another implementation lane. No
automatic resolution of its archive against these records is performed.
The later `d9490141` PR file inventory lists seven modified paths, including
the Bayesian source, active plan/specification and shared files, with no archive
paths in that inventory. That later proposal is not tested or accepted here;
the integrator must reconcile its actual final tree.

## Completed implementation preserved in the patch

The patch applies to `b344f379926e934be30edc0842779dc5e3455f86`. Its SHA-256 is:

```text
6e54a27ba271c1af05539773048dc10e020b4abde2e0340505ed8aa3d6999767
```

[candidate-source.json](candidate-source.json) records every candidate source,
test and fixture digest plus the six patch paths. The original independent
fixture and bootstrap preservation tests already exist in the baseline commit.

- Replace the private 48-step normal-quantile bisection with
  `Normal::standard().inverse_cdf`.
- Preserve unnormalised Gaussian prior/likelihood arithmetic, deterministic RNG,
  average tied ranks, signed-zero equivalence, odd-middle exclusion, diagnostic
  formulas/tolerances and empirical type-7 bootstrap intervals.
- Repair exact-constant split rounding with equality guards. Every split must
  be constant; one varying split reaches the original estimator. R-hat checks
  nonfinite moments first, retaining its previous overflow error.
- Add dedicated independent quantile, domain, tail, score, constant/edge and
  public-behavior tests. Bootstrap has no analytic normal/Student-t helper to
  replace; its production arithmetic remains unchanged.

The quantile's sole caller uses validated chains with at most 160,000 pooled
values. Blom probabilities are in `[0.625/160000.25, 1-0.625/160000.25]`, strictly
within `(0,1)`. Legacy impossible-input saturation is characterized separately;
no new public quantile API is introduced.

The candidate initially caused the existing 80-draw constant-chain ESS test to
return `2.0253164556962027` instead of null. The exact-equality repair fixes that
regression and some pre-existing long-constant-chain artifacts. The
[before-repair probe](constant-split-probe-before-repair.json) distinguishes
those categories; the [red test](constant-split-red.log) records the actual
half-40 failure. No blanket bitwise-preservation claim is made.

## Independent fixtures and actual qualification

The 83-row mpmath 1.3.0 fixture uses 400 decimal digits and exact binary64 inputs.
It was committed before migration and still regenerates byte-for-byte:

```text
normal.json: fd591fe542d0c13644f887941ba74c8d6381012f730bbb7f84d8926af79418ac
generate.py: edc9be338d189edefacdf7c18829ba2f9e5d4d733cd48f4648b306313ef1fc8a
```

The five original-main behavioral cases are a separate replay baseline, not
independent mathematical truth. Their request/response values remain unchanged;
later provenance adds the target, libc and appropriate existing grid tolerances.

The final [qualification receipt](qualification.json) binds commands, logs,
source hashes, toolchain and actual outcomes to the reconstructed patch plus
the candidate dependency overlay. [Candidate review](candidate-review.md)
records the independent Conductor source review and applied findings.

Reconstruction from the baseline and the two checked patches produced
**393 passed, one failed and one inherited ignored test across 28 core targets**.
The sole failure is the CDF/survival gate below. Clippy with warnings denied,
scoped rustfmt, oracle regeneration and repository policy pass. The scientific
review freshness check fails pending the required re-review.

The unchanged normal-distribution gate has these results:

| Comparison group | Result |
| --- | --- |
| Inverse CDF | All 22 reference cases pass; absolute tolerance `2e-10` |
| Normalized log PDF | All 18 cases pass; absolute `2e-13` plus relative `5e-15` |
| CDF and survival | 14 of 36 comparisons fail; absolute zero plus relative `2e-13` |
| Relative Gaussian scores | Original arithmetic passes; zero absolute plus relative `5e-15` |

All 36 CDF/survival comparisons execute. Seven CDF and seven survival failures
include central and tail values. At `x=-10`, statrs returns
`7.61985302455964105e-24` versus reference `7.61985302416052545e-24`, approximately
`5.2e-11` relative error. The active candidate gate remains failing; no expected
value, tolerance, test selection or ignore was changed to make it pass.

Two isolated negative controls are detected: reversed quantile probability and
normalized-log-density subtraction. The latter loses the near-centre relative
score `-5e-21`. A stale initial mutation-build invocation is excluded; recorded
logs show actual recompilation and failure. Two negative controls do not
establish a mutation score, branch coverage or overall coverage.

## Exact shared requests and reconciliation differences

The dependency request is:

```toml
statrs = { version = "=0.19.1", default-features = false, features = ["std"] }
```

Registry archive SHA-256, checked against the downloaded archive and lock entry:

```text
c1837f9a1bd13dabbdbb04489c39451482d3cdd35a6e1a0d0f3570dea2b8c8be
```

[statrs-dependency-request.patch](statrs-dependency-request.patch) and
[dependency-request.json](dependency-request.json) give the exact original-base
manifest/lock proposal, all package checksums and lock digests. This patch is
**not** rebased onto current main or #625. The integrator reconciles it with
the merged linalg dependency graph; do not overwrite the newer shared lock.

Cargo adds 15 lock packages, removes none and changes only the existing
`innovate-core` package entry. Optional nalgebra/glam packages remain in the
resolver's lock closure while compiled statrs features are only `std`. Do not
trim those entries or infer enabled features from lock presence. The complete
workspace feature-tree attempt stopped on uncached `argminmax`, and
`cargo-audit` was unavailable: no full dependency graph/audit pass is claimed.
Retain `libm` pending a separate whole-crate consumer audit.

| Decision | Preserved candidate | #625 source observed at the named head |
| --- | --- | --- |
| Gaussian scores | Retain original unnormalised arithmetic with near-centre/extreme-scale references | Replace with normalized `ln_pdf` calls |
| ESS/R-hat constants | Exact-constant guards; other moment arithmetic retained | Change ESS centering generally |
| statrs features | Defaults disabled; `std` only | `statrs = "=0.19.1"` with defaults |
| Track status | Ownership/shared integration/CDF/scientific holds open | Proposes archival |

This packet has not executed #625. These are source-level reconciliation
differences, not a claim that its entire suite was reproduced. Its declared
passes do not establish the independent gates above.

The changed Bayesian source also invalidates
`docs/contracts/scientific-pedagogical-review.json`: the unchanged contract check
fails with a required re-review. The frozen candidate's Bayesian source hash is
`b83e0abf3f400c77bd441c7feb1f487bb1b027dd32578660439c771ac5c4e99f`.
The integrator/documentation owner must review the accepted implementation and
refresh its exact fingerprint afterward. No fingerprint is advanced here to
bypass that review. Causal, shared manifests/locks, CI and the common registry
remain untouched by this lane.

## Reproduction and evidence limits

Check disk, memory and active work before builds. This run used its own `/tmp`
checkout/cache/target because the original filesystem had about 6 MB free.
Rust/Cargo 1.99.1, two Cargo jobs and disabled incremental compilation were used;
no other writer's work, cache or files were deleted or altered.

For the preserved candidate only, create a disposable copy of baseline commit
`b344f379926e934be30edc0842779dc5e3455f86`. Preserve the core integration tests'
three compile-time fixtures under the scientific ecosystem alignment track.
Apply the two checked patches to that copy, verify every digest in
`candidate-source.json` and the candidate lock digest, then run:

```sh
git apply --check candidate-implementation.patch
git apply candidate-implementation.patch
git apply --check statrs-dependency-request.patch
git apply statrs-dependency-request.patch
cargo test --locked --offline -p innovate-core --no-fail-fast
cargo clippy --locked --offline -p innovate-core --lib --tests -- -D warnings
```

The patch arguments denote the files from this packet; use their absolute paths
when the working directory is the disposable checkout. Offline operation needs
the complete locked cache. The expected candidate result still includes the
CDF gate failure. The one inherited live-OECD test is ignored upstream; this
lane introduced no new ignored test.

Python 3.12.14 was used only for the independent oracle and policy checks, not
Python 3.15.2 installed-consumer qualification. Required integrated-head CI,
full workspace, installed Python/Julia/R consumers, clean-package/platform,
security, documentation and scientific acceptance remain open. Retain >80%
overall and >90% critical-module coverage, >70% native mutation score and weekly
schedule, and reproducible performance requirements.

All 38 scientific obligations, the four historical human scenario checks,
fail-closed release gate, offline-security qualification and planning-only
downstream restrictions remain. No NUTS, linalg, merge, release/publication,
permission/account change, gate weakening, upstream write or unverified
deletion is performed by this packet.
