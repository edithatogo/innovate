# Compare the native search with R ML initialized near the generating roots.
# Positive gaps can indicate that the single R start missed a better basin.
for (seed in c(17L, 42L, 123L, 987L)) {
  for (sign in c(-1, 1)) {
    set.seed(seed)
    ar <- sign * 0.98
    ma <- -sign * 0.96
    values <- as.numeric(stats::arima.sim(
      list(ar = ar, ma = ma), n = 500L, n.start = 1000L
    ))
    order <- c(1L, 0L, 1L)
    native <- innovate.R::kernel_call("preprocess_series", payload = list(
      method = "sarima_fit", values = values, order = order, seasonal_order = integer(4)
    ))$result
    reference <- stats::arima(
      values, order = order, method = "ML", include.mean = FALSE,
      transform.pars = FALSE, init = c(ar, ma),
      optim.control = list(maxit = 2000L, reltol = 1e-12, ndeps = rep(1e-6, 2))
    )
    stopifnot(reference$code == 0L, is.finite(native$log_likelihood))
    gap <- native$log_likelihood - as.numeric(stats::logLik(reference))
    cat(sprintf("seed=%d sign=%+d native-minus-reference=%.9g\n", seed, sign, gap))
  }
}
