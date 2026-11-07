set.seed(593)
order <- c(1L, 0L, 1L)
seasonal <- c(1L, 0L, 1L, 4L)
for (root in c(0.97, -0.97)) {
  values <- as.numeric(stats::arima.sim(
    list(ar = c(0.35, 0, 0, root, -0.35 * root), ma = c(-0.2, 0, 0, -0.9, 0.18)),
    n = 320L, n.start = 1000L
  ))
  reference <- stats::arima(values, order = order,
    seasonal = list(order = seasonal[1:3], period = seasonal[4]),
    method = "ML", include.mean = FALSE)
  native <- innovate.R::kernel_call("preprocess_series", payload = list(
    method = "sarima_fit", values = values, order = order, seasonal_order = seasonal
  ))$result
  stopifnot(reference$code == 0L, length(native$fitted) == length(values))
  gain <- native$log_likelihood - as.numeric(stats::logLik(reference))
  coefficient_error <- max(abs(unlist(native$coefficients) - stats::coef(reference)))
  cat(sprintf("Seasonal AR %+.2f: Rust minus R likelihood %.9g, max coefficient error %.9g\n",
    root, gain, coefficient_error))
}
