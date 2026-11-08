for (ma in c(0.98, -0.98)) {
  set.seed(942)
  values <- as.numeric(stats::arima.sim(
    list(ar = 0.35, ma = ma), n = 600L, n.start = 200L
  ))
  order <- c(1L, 0L, 1L)
  reference <- stats::arima(values, order = order, method = "ML",
    include.mean = FALSE, optim.control = list(reltol = 1e-12, ndeps = rep(1e-6, 2)))
  native <- innovate.R::kernel_call("preprocess_series", payload = list(
    method = "sarima_fit", values = values, order = order, seasonal_order = integer(4)
  ))$result
  gain <- native$log_likelihood - as.numeric(stats::logLik(reference))
  error <- max(abs(unlist(native$coefficients) - stats::coef(reference)))
  cat(sprintf("ARMA(1,1) MA %+.2f: likelihood gap %.9g, coefficient error %.9g\n",
    ma, gain, error))
  stopifnot(reference$code == 0L, abs(gain) < 1e-6, error < 5e-6)
}
