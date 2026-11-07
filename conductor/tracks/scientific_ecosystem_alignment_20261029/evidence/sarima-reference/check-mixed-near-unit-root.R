set.seed(328)
for (root in c(0.985, -0.985)) {
  values <- as.numeric(stats::arima.sim(list(ar = root, ma = 0.6), n = 300L, n.start = 800L))
  order <- c(1L, 0L, 1L)
  reference <- stats::arima(values, order = order, method = "ML", include.mean = FALSE)
  native <- innovate.R::kernel_call("preprocess_series", payload = list(
    method = "sarima_fit", values = values, order = order, seasonal_order = integer(4)
  ))$result
  gain <- native$log_likelihood - as.numeric(stats::logLik(reference))
  cat(sprintf("ARMA(1,1) AR %+.3f: Rust minus R likelihood %.9g\n", root, gain))
  stopifnot(reference$code == 0L, length(native$fitted) == length(values), abs(gain) < 1e-6)
}
