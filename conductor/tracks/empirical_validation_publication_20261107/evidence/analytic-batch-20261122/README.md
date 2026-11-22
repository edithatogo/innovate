# Analytic prediction batch

On 22 November 2026, the macOS ARM64 wheel from signed release run
`44996337632` was installed in a fresh Python 3.15.2 environment. GitHub's
attestation verifier authenticated the receipt and its source commit
`f0f31f5a3e4f1ae10d3c8c3ea1f3308c9e0d6823`. The wheel SHA-256 was
`93caf0e5db5d7e1efc3350f63da56a249f9871a190172bd40c0be14be6d22559`.

The batch checked 540 predicted values across 90 parameter combinations for
Bass, Logistic, Gompertz and Fisher–Pry curves. Expected values use Python's
70-digit `Decimal` exponential on the exact binary64 inputs. Each prediction
used absolute tolerance `1e-10` and relative tolerance `1e-12`. The Polars
frames retained their time, identifier and text columns and row order. Four
invalid parameter combinations returned errors. The exact counts and largest
absolute errors are in [results.json](results.json).

This scope covers deterministic levels on synthetic finite inputs. Fit,
uncertainty, empirical validation and the complete release obligations remain
open. The wheel's signed source inventory SHA-256 is
`1a9f5541b60906f286b7fda765d23757c8bd9c99f06ad46536c3ab8194b538bc`.
The signed receipt is retained with the [three-platform wheel evidence](../signed-wheels-20261122/README.md).

The [checker](check.py) has SHA-256
`74329cc2afdcff7ad6d65e7b4ccd7e4fd997f10d7db5bffe20c71ae8a52e12b7`.
The result has SHA-256
`fd12baf5b417cfaf8e4b7c53cff56b7ed7bb6bc89c7596dece963ec0ea0deb46`.
To replay, install the verified wheel into a fresh Python 3.15.2 environment,
change to a directory outside the checkout containing the wheel, and run the
checker with that environment's Python in isolated mode. Capture its JSON
output and compare the reported wheel hash and results.
