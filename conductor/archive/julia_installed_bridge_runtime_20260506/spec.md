# Specification: Julia Installed-Package Bridge Runtime Readiness

## Overview

Make the Julia binding work in an installed-package context instead of only in
a repository checkout. The Julia package should continue to be thin, but it
must stop assuming the shared Python source tree is always adjacent to the Julia
package source.

## Functional Requirements

1. Detect repository checkout mode separately from installed-package mode.
2. Keep checkout-mode development working with the existing Python bridge path.
3. Allow installed-package bridge calls to run without depending on
   `../../../src/innovate/kernel.py`.
4. Add a smoke test for installed-package runtime behavior.
5. Update Julia docs and CI/publish workflow steps to reflect the installed
   package path.

## Non-Functional Requirements

1. The fix must not change the kernel contract or payload shapes.
2. The installed-package path must remain a thin bridge over the Python kernel.
3. The behavior should be easy to validate in CI with minimal churn.

## Acceptance Criteria

1. Julia bridge calls work without a repo checkout when `INNOVATE_PYTHON_COMMAND`
   points to a Python environment that has `innovate` installed.
2. Checkout-mode tests still pass unchanged.
3. The registry-readiness docs describe the installed-package runtime contract
   without referring to a checkout-only blocker.
4. CI or publish workflows include an installed-package smoke validation.

## Out of Scope

1. Rewriting the Python kernel bridge.
2. Adding Julia-native model logic.
3. Publishing to Julia General in this track.
