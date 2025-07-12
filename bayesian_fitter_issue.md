# BayesianFitter Segmentation Fault

This document details a segmentation fault encountered when running tests for the `BayesianFitter`.

## Problem Description

The tests for the `BayesianFitter` are consistently failing with a `Fatal Python error: Segmentation fault`. This error occurs both with multiprocessing (`chains > 1`) and with a single chain (`chains=1`).

## Steps to Reproduce

1.  Run the tests for the `BayesianFitter`:
    ```bash
    pytest tests/test_bayesian_fitter.py
    ```

## Error Messages

The primary error is a segmentation fault, but it is often preceded by other errors, such as `ConnectionResetError` when running with multiple chains.

**With `chains=2`:**
```
ConnectionResetError: [Errno 104] Connection reset by peer
...
Fatal Python error: Segmentation fault
```

**With `chains=1`:**
```
Fatal Python error: Segmentation fault
```

## Debugging Steps Taken

1.  **Fixed `TypeError` in `differential_equation`:** The initial tests failed because the `differential_equation` methods in the models had a signature that was incompatible with `pymc.ode.DifferentialEquation`. This was fixed by creating a wrapper function.
2.  **Fixed `TypeError` with symbolic boolean evaluation:** The tests then failed because the `differential_equation` methods were using standard Python `if` statements on symbolic `TensorVariable` objects. This was fixed by using `pytensor.tensor.switch`.
3.  **Disabled multiprocessing:** The tests were run with `chains=1` to rule out any issues with multiprocessing. The segmentation fault still occurred.

## Next Steps

1.  Create a minimal, self-contained Python script to reproduce the error without the `innovate` library.
2.  If the error can be reproduced, report the issue to the `pymc` or `pytensor` developers.
