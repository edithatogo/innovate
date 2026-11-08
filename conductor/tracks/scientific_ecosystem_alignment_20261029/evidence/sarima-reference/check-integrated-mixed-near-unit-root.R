for (root in c(0.98, -0.98)) {
  set.seed(811)
  increments <- as.numeric(stats::arima.sim(
    list(ar = root, ma = 0.55), n = 400L, n.start = 1000L
  ))
  values <- cumsum(c(0, increments))
  # Rust conditions its likelihood on the initial observation. Fit R to
  # the observed differences to use the same likelihood definition.
  reference <- stats::arima(diff(values), order = c(1L, 0L, 1L),
    method = "ML", include.mean = FALSE)
  native <- innovate.R::kernel_call("preprocess_series", payload = list(
    method = "sarima_fit", values = values, order = c(1L, 1L, 1L),
    seasonal_order = integer(4)
  ))$result
  gain <- native$log_likelihood - as.numeric(stats::logLik(reference))
  error <- max(abs(unlist(native$coefficients) - stats::coef(reference)))
  cat(sprintf("Integrated ARMA AR %+.2f: likelihood gap %.9g, coefficient error %.9g\n",
    root, gain, error))
  stopifnot(reference$code == 0L, length(native$fitted) == length(values),
    abs(gain) < 1e-6, error < 5e-6)
}
