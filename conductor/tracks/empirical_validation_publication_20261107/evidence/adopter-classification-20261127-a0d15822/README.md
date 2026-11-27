# Adopter classification qualification on reconciled source

The replay used release source `a0d158224eb70e25cabebceed5cafd9718436cf8`.
It regenerated the exact-rational references from the
[20 November qualification](../adopter-classification-20261120/README.md).
The generated cases had the same SHA-256 as the earlier corpus.

Each macOS ARM64 binding passed 1,101 independent reference cases, six
invalid-input cases and an independently calculated fitted Logistic case.
The [Python log](python.log) came from the installed, verified signed
wheel. The [Julia log](julia.log) came from the bundled local library.
The [R log](r.log) came from a separate installation of the binary archive.
Julia and R loaded the library built from this source with
`rustup run 1.100.0 cargo build --locked --release --offline -p innovate-ffi -j 2`.

The [native report](native.json) binds the harnesses, artifacts and logs
to the release source. The [scientific report](scientific.json) covers
the sample-weighted label classifier and its supported fitted single-rate
adapter. The hosted build exercised the bundled Julia and R packages on
all three supported platforms. The separately signed wheel receipts cover
179 installed-wheel tests and 42 Arrow tests per platform with zero skips.

[Signed wheel evidence](https://github.com/edithatogo/innovate/actions/runs/45891650893) ·
[Hosted native and binding checks](https://github.com/edithatogo/innovate/actions/runs/45891480737)

The remaining ledger obligations and aggregate release attestation are open.
