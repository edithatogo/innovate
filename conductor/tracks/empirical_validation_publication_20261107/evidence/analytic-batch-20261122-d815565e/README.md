# Analytic replay on the combined-source wheel

The existing [analytic batch](../analytic-batch-20261122/README.md) was replayed
on the macOS ARM64 wheel signed in
[release run 45003005112](https://github.com/edithatogo/innovate/actions/runs/45003005112).
The [checker](check.py) verifies the wheel SHA-256 and installed Python and
native-extension bytes before comparing 540 predictions for Bass, Logistic,
Gompertz and Fisher–Pry against 70-digit Decimal references. Polars input
columns and row order were preserved, and four invalid parameter cases were
rejected. [Results](results.json) matched the earlier independent batch.

The checker SHA-256 is
`b96eb7d96982f0cc4610a91c56d71253d01ee9f4424adf7644ea6730eef2e05a`.
The results SHA-256 is
`5798f2a37df0768f679cf75e817f4f606f03a37cbf9f16c688d6bc8265d7122d`.
The signed wheel receipt is retained with the
[combined-source wheel evidence](../signed-wheels-20261122-d815565e/README.md).
These bounded deterministic results do not qualify fitting, uncertainty,
empirical validation or the full release obligations.
