"""Generate and check a historical mixed SARIMA fit in separate environments."""

import json
import sys

mode, path = sys.argv[1:]
if mode == "generate":
    import numpy as np
    from statsmodels.tsa.arima_process import arma_generate_sample
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rng = np.random.default_rng(42)
    stationary = arma_generate_sample(
        [1, -0.45, 0, 0, -0.35, 0.1575],
        [1, -0.2, 0, 0, 0.3, -0.06],
        400, burnin=100, distrvs=rng.standard_normal,
    )
    seasonal = np.zeros(404)
    for i, value in enumerate(stationary, start=4):
        seasonal[i] = seasonal[i - 4] + value
    values = np.cumsum(np.r_[0, seasonal])
    fit = SARIMAX(values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4),
                  concentrate_scale=True).fit(disp=False)
    assert fit.mle_retvals["converged"]
    with open(path, "w", encoding="utf-8") as output:
        json.dump(dict(values=values.tolist(), coefficients=fit.params.tolist(),
                       fitted=fit.fittedvalues.tolist(), log_likelihood=fit.llf),
                  output, allow_nan=False)
elif mode == "check":
    import polars as pl
    from innovate import execute, sarima_fit

    with open(path, encoding="utf-8") as source:
        reference = json.load(source)
    options = dict(order=(1, 1, 1), seasonal_order=(1, 1, 1, 4), historical=True)
    native = execute("preprocess_series", method="sarima_fit",
                     values=reference["values"], **options)["result"]
    fitted = sarima_fit(pl.Series(reference["values"]), **options)
    coefficient_error = max(abs(x - y) for x, y in
                            zip(native["coefficients"], reference["coefficients"], strict=True))
    fitted_error = max(abs(x - y) for x, y in zip(fitted, reference["fitted"], strict=True))
    likelihood_error = abs(native["log_likelihood"] - reference["log_likelihood"])
    print(f"coefficient {coefficient_error:.12g}\nfitted {fitted_error:.12g}\n"
          f"log likelihood {likelihood_error:.12g}")
    assert coefficient_error < 2e-4 and fitted_error < 2e-4 and likelihood_error < 1e-6
else:
    raise ValueError(f"unknown mode {mode}")
