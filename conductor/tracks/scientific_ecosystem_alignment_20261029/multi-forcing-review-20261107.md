# Multi product covariate forcing review

Named additive p/q/m and interaction coefficients are supported through the
existing shared history path. The default corrected layout follows the parameter
names. An explicit historical layout preserves the pre-migration product-triple
indexing mismatch. Independent SciPy DOP853 fixtures reproduce both layouts,
including signed effective coefficients, instantaneous rates and endpoint-held
forecasts. Effective nonpositive capacity gives zero growth for that product.

The perturbed-start fit initially passed the training residual check but failed
the original held-out forecast tolerance. The retained failure exposed premature
absolute objective-improvement termination near zero. The shared least-squares
solver now compares improvement with the previous objective. Its gradient,
iteration and residual criteria remain in effect. The forecast regression passes
at its original tolerance. Signed beta parameters scale with their corresponding
base parameter's units.

The full Rust core/C ABI and Polars suites pass because the optimizer is shared
across model families. Installed Python exercises both layouts, Polars fitting
and the normal fitted forecast method. Julia and R reproduce the same layouts,
fit and saved-state forecast. Local package metadata, wheel contents and exact
Rust module/test/fixture inclusion pass.

This checkpoint requires explicit forcing-fit initial parameters and conditions
on known controls and initial state. It does not close automatic initialization,
events, calibrated uncertainty, global identifiability, branch/mutation or
platform/scientific release obligations. Existing local identifiability reports
retain their limited interpretation.
