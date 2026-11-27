# Signed installed-wheel evidence

Release workflow run `44996337632` built wheels on Linux x86-64, macOS ARM64
and Windows x86-64 from source commit
`f0f31f5a3e4f1ae10d3c8c3ea1f3308c9e0d6823`. The signed source inventory is
`1a9f5541b60906f286b7fda765d23757c8bd9c99f06ad46536c3ab8194b538bc`.
Each wheel ran 164 minimal and 42 Arrow cases with zero skips, including the
live OECD case and the adopter-classification reference. The workflow and
job outcomes are available in the [release run](https://github.com/edithatogo/innovate/actions/runs/44996337632).

Each `receipt-TARGET.json` is accompanied by `signed-TARGET.json`, a GitHub
SLSA provenance bundle. The two XML files retain the original minimal and
Arrow test reports. Their hashes and the wheel hash are recorded in the
corresponding signed receipt. The verifier checked the signer workflow,
signer and source commit, artifact archive and wheel hashes, report hashes,
target, Python version, test identities and zero skipped cases.

The signed results cover the installed-wheel cases. The release readiness
job in the same run reported 37 incomplete obligations. Complete native,
scientific and release-manifest evidence remains necessary for promotion.

For a target, verify `receipt-TARGET.json` with `gh attestation verify`,
using `signed-TARGET.json` as the bundle, repository `edithatogo/innovate`,
signer workflow `edithatogo/innovate/.github/workflows/release-readiness.yml`,
and the source commit above for both signer and source digest. Wheel archives
and source distributions are retained as artifacts of the linked run.
