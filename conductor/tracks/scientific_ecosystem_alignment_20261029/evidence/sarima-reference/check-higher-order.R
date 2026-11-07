set.seed(51)
for (order in list(c(2L, 0L, 1L), c(1L, 0L, 2L), c(2L, 0L, 2L))) {
  values <- as.numeric(stats::arima.sim(list(ar = c(0.5, -0.3), ma = c(0.3, -0.15)), n = 200))
  reference <- stats::arima(values, order = order, method = "ML", include.mean = FALSE)
  native <- innovate.R::kernel_call("preprocess_series", payload = list(
    method = "sarima_fit", values = values, order = order, seasonal_order = integer(4)
  ))$result
  gain <- native$log_likelihood - as.numeric(stats::logLik(reference))
  cat(paste(order, collapse = "/"), sprintf("native minus R likelihood %.12g\n", gain))
  stopifnot(reference$code == 0L, length(native$fitted) == length(values), gain > -1e-6)
}
