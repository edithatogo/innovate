# Analytic model references

These level predictions use the equations below and fixed synthetic inputs.
Their values were calculated independently of the Rust implementation.
Times, parameters, options, expected columns and absolute tolerances are
recorded in the [manifest](manifest.json). Project-owned material is
Apache-2.0. [Licence](../../LICENSE)

## Bass without imitation

For zero imitation and initially zero cumulative adoption, the unadopted
population solves a first-order exponential decay. The level at time `t`
is `m * (1 - exp(-p*t))`. The fixture starts at zero and checks the level
after five time units.

## Logistic midpoint

The level is `L / (1 + exp(-k*(t-x0)))`. At `t = x0` the denominator
is exactly two. The fixture also evaluates the same formula before the
midpoint.

## Gompertz unit displacement

The level is `a * exp(-b * exp(-c*t))`. With `b = 1` the initial
level is `a / e`. A positive time checks the interior of the curve.

## Fisher–Pry substitution midpoint

For the natural-log-odds convention in the public model, the share is
`1 / (1 + exp(-alpha*(t-t0)))`. At `t = t0` the share is one half.
Times equally spaced on either side have shares that sum to one.

## Norton–Bass generation launch

For elapsed time `u > 0`, the adoption fraction for one generation is
`F(u) = (1 - exp(-(p+q)*u)) / (1 + (q/p)*exp(-(p+q)*u))`. Set the
fraction to zero up to and including the launch at `u = 0`.
With `F1` and `F2` measured from their respective launch times, the two
successive-generation levels are `m1*F1*(1-F2)` and
`(m2 + m1*F1)*F2`. The fixture covers both sides of the second launch.

## Competition with zero adoption rates

Set innovation, imitation and interaction coefficients to zero. For each
product the time derivative is then zero, so both levels stay equal to
their distinct starting values. The fixture checks column order and the
levels at the start and after five time units.

These cases cover deterministic levels at the declared inputs. They provide
no fit, uncertainty, observed-data, or full scientific release acceptance.
