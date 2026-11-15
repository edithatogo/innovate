# Independent statistical references

These synthetic references exercise the Track 5 distribution migration. They
are generated from analytic definitions using **mpmath 1.3.0 at 400 decimal
digits**. The generator imports neither Innovate nor Rust, libm, statrs or SciPy.
It is a test oracle, with no role in production computation.

The initial generation used CPython 3.12.14 in an isolated reference environment.
The generator is compatible with Python 3.10 and newer; the required mpmath
version is enforced. This reference environment does not qualify the repository's
separate Python 3.15.2 installed-consumer gate.

## Reproduction and provenance

From this directory, with mpmath 1.3.0 available:

```sh
python generate.py
python generate.py --check
```

`--check` recomputes every reference and requires byte-for-byte equality with
`normal.json`. The JSON records the original main commit, generator SHA-256,
input SHA-256, precision and conversion conventions. The command prints the
complete fixture's SHA-256. There are no timestamps or platform-dependent paths
in generated output.

Inputs are first represented as binary64 and then converted exactly to mpmath.
For example, `1 - 1e-6` is the actual binary64 input that Rust receives; it is
not silently replaced with the ideal decimal 0.999999. Each quantile also
records the hexadecimal representation of its input. Expected results retain
65 significant reference digits and a single final rounding to binary64.
Nonfinite values use the JSON strings `"-inf"`, `"inf"` and `"nan"`.

## Equations and coverage

For the standard normal distribution:

\[
Q(p)=\sqrt{2}\,\operatorname{erf}^{-1}(2p-1),\qquad
F(x)=\tfrac12\operatorname{erfc}(-x/\sqrt{2}),\qquad
S(x)=\tfrac12\operatorname{erfc}(x/\sqrt{2}).
\]

For a Gaussian with finite mean and positive finite standard deviation:

\[
\log f(x;\mu,\sigma)=-\tfrac12[(x-\mu)/\sigma]^2
-\log\sigma-\tfrac12\log(2\pi).
\]

The separate `gaussian_relative_log_density` group evaluates only
\(-\tfrac12[(x-\mu)/\sigma]^2\), matching the existing posterior's intentional
omission of constants. Those constants cancel when comparing predictions with
fixed variance or fixed Gaussian priors; they must not be confused with a
normalized density reported to a caller.

The fixture covers the five probabilities specified by the track, the extreme
reachable Blom ranks for 160,000 pooled draws, additional interior and remote
tails, mathematical quantile endpoints, CDF/survival underflow and saturation,
shifted and scaled densities, finite and overflowing large negative log
densities, and nonfinite evaluation arguments. Survival references are retained
separately so a saturated upper CDF cannot hide a missing right tail.

## Tolerances and contract boundaries

Tolerance fields were fixed before migration. Quantile references use an
absolute tolerance of `2e-10`; this permits the legacy 48-step bisection's
rounding at the required `1 - 1e-6` tail while remaining tighter than the
repository's general well-scaled tolerance. Existing repository diagnostic
tolerances remain unchanged. `legacy_domain: true` selects the interval
`[1e-6, 1 - 1e-6]`, which contains every reachable Blom rank. Rows outside that
interval qualify the distribution dependency only; they do not expand the
private rank-transform contract. In particular, the most extreme mathematical
quantiles exceed the old `[-8, 8]` search interval.

CDF and survival comparisons use zero absolute tolerance and `2e-13` relative
tolerance, with exact comparisons at the center. This prevents zero from passing
as a tiny but representable tail probability. Mathematical underflow to
binary64 zero and infinite limits remain exact classification checks.
Normalized log-PDF references use `2e-13` absolute plus `5e-15` relative
tolerance. Relative log densities use zero absolute plus `5e-15` relative
tolerance: the small negative values for residuals `1e-10` and `1e-8` must
survive. Subtracting `Normal.ln_pdf(0)` from `Normal.ln_pdf(z)` can erase those
values through cancellation and is not a behavior-preserving substitution.

`legacy_quantile_boundaries` characterizes unreachable private inputs separately.
The historical bisection gives `-8 + 2^-45` or `8 - 2^-45` after 48 partitions
at out-of-domain values and endpoints; NaN follows the lower branch. These are
closed-form binary-partition observations, **not normal-distribution quantiles**.
They are included to make any intentional change visible, not to promote invalid
probabilities or this accidental private behavior into a public API.

No reference establishes empirical validity, sampler convergence, NUTS readiness,
causal identification, downstream implementation permission, or release approval.

## Primary references

- [mpmath inverse error function and error-function definitions](https://mpmath.org/doc/1.3.0/functions/expintegrals.html#erfinv)
- [mpmath number types and precision](https://mpmath.org/doc/1.3.0/basics.html#number-types)
