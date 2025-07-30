# ODE Solver Instability with CompetitionModel

## Problem

During the implementation of the `CompetitionModel` (a generalized multi-product diffusion model), we encountered persistent, unrecoverable errors (`Segmentation fault`, `Fatal Python error: Aborted`) during the model fitting process.

## Debugging Steps Taken

1.  **Initial Implementation:** A `CompetitionModel` was implemented in `src/innovate/compete/competition.py`, consolidating previous work and adding support for an arbitrary number of products.
2.  **Testing:** A new test suite, `tests/test_competition.py`, was created.
3.  **Initial Failures:** The tests initially failed with `ValueError` and `lsoda` solver warnings, indicating that the `scipy.integrate.solve_ivp` function was returning truncated results.
4.  **Solver Adjustments:** The `solve_ivp` call was made more robust by increasing error tolerances (`rtol`, `atol`). This resolved the `ValueError` but resulted in a poor model fit (`R² < 0.5`).
5.  **Improved Initial Guesses:** The `initial_guesses` method was improved to provide more realistic starting parameters for the optimizer.
6.  **Optimizer Change:** After the improved guesses still led to segmentation faults with complex models (especially those with covariates), the optimizer was switched from `scipy.optimize.minimize(method='L-BFGS-B')` to the more robust `scipy.optimize.differential_evolution`.
7.  **Persistent Crashes:** Despite all these changes, the test suite continued to crash with fatal, low-level errors originating from the numerical libraries (`numpy`, `scipy`).

## Diagnosis

The root cause appears to be a fundamental instability in the numerical environment, likely an interaction between the installed versions of `numpy`, `scipy`, and other C-backed libraries. The `CompetitionModel`, with its large number of parameters and complex, coupled differential equations, is too demanding for the current `LSODA` solver in this environment, pushing it into an unstable state that it cannot recover from.

## Proposed Solution

As per the project's `roadmap.md`, the long-term solution is to implement a high-performance JAX backend using a more advanced ODE solver.

**Action Plan:**
1.  **Defer Implementation:** Pause the development of the `CompetitionModel`.
2.  **Prioritize JAX Backend:** Proceed with **Phase 6** of the roadmap to implement the `JaxBackend` and integrate the `diffrax` library.
3.  **Re-implement with `diffrax`:** Once the new backend is in place, re-implement the `CompetitionModel` using the `diffrax` solver, which is designed for better performance and stability with stiff and complex ODEs.
4.  **Revert Code:** For now, revert the `src/innovate/compete` module to its previous stable state to ensure the repository is not left in a broken state.
