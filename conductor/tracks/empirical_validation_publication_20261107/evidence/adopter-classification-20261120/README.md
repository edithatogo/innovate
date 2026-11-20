# Adopter-classification qualification, 20 November 2026

The qualified behavior assigns the five historical labels from sample-weighted
time moments. It covers finite, strictly increasing times, nonnegative
weights with positive mass, inclusive boundary ties and the existing fitted
single-rate adapter. These labels describe model samples.

[Scientific results](scientific.json) record 1,101 direct reference cases and
one fitted Logistic case. The generator uses exact rational arithmetic on
the input binary64 values to calculate the mean and centered variance.
It repeats the 765 heterogeneous-weight cases in the installed wheel suite
and adds 336 cases with one to 65 samples on signed and irregular grids.
The additional cases span subnormal through very large finite values. Julia
and R reconstruct the inputs from their bits before testing the public
binding. The fitted case uses independent analytic levels and rates and
checks the predicted rates within an absolute tolerance of `1e-8`.

[Native results](native.json) record 1,108 cases on each local binding. Each
run includes six invalid-input cases and the fitted Logistic case. Python
compares the returned labels through Polars and JSON and checks that the
DataFrame preserves input rows. The local Julia package loads its bundled
release dylib. R runs from the separately installed binary archive built
with that same dylib. [Python output](python.log), [Julia output](julia.log),
[R output](r.log)

The verified GitHub signatures cover the installed Python wheel suite on all
three supported platforms, including its 765 reference cases. They bind
`c1399484aa7efa224304b0a84c559ecffc10726e`, which has the same release source
digest as the local build at `6feb3e509d860442d884534fed86215365af0f6f`. The
Julia and R hosted jobs also built and exercised their bundled release
library on all three platforms. The additional 336 reference cases and the
new fitted fixture were checked locally. [Signed wheel results](https://github.com/edithatogo/innovate/actions/runs/44608324560),
[Hosted binding checks](https://github.com/edithatogo/innovate/actions/runs/44613411217)

## Replay

From the repository root, verify the release source digest against
`scientific.json` with `scripts/check_migration.py --source-digest`. Use
the selected Python 3.15.2 environment with the source-built wheel, Polars
and the locked test dependencies. Build the Julia and R bundles with
`cargo build --release --locked -p innovate-ffi` and stage the native library
as the CI binding job does. Install the R archive and select its library.
Use the Julia project with the staged library. Leave the
`INNOVATE_NATIVE_LIBRARY` override unset.

In the following commands, set `qualification` to this evidence directory,
`qualification_python` to the selected Python 3.15.2 interpreter and
`output` to a writable directory for the generated cases.

```sh
"$qualification_python" -B "$qualification/generate_cases.py" "$PWD" "$output/adopter-cases.json"
"$qualification_python" -B "$qualification/check_adopters.py" "$output/adopter-cases.json"
julia --startup-file=no --project=bindings/julia \
  "$qualification/check_adopters.jl" "$output/adopter-cases.json"
Rscript --vanilla "$qualification/check_adopters.R" "$output/adopter-cases.json"
```
