set.seed(142)
for (term in c("ar", "ma")) {
  order <- if (term == "ar") c(2L, 0L, 0L) else c(0L, 0L, 2L)
  for (root in c(0.985, -0.985)) {
    ar <- c(root + 0.25, -root * 0.25)
    coefficients <- if (term == "ar") ar else -ar
    values <- as.numeric(stats::arima.sim(setNames(list(coefficients), term), n = 300L, n.start = 800L))
    reference <- stats::arima(values, order = order, method = "ML", include.mean = FALSE)
    native <- innovate.R::kernel_call("preprocess_series", payload = list(
      method = "sarima_fit", values = values, order = order, seasonal_order = integer(4)
    ))$result
    gain <- native$log_likelihood - as.numeric(stats::logLik(reference))
    cat(sprintf("%s(2) %+.3f: Rust minus R likelihood %.9g\n", term, root, gain))
    stopifnot(reference$code == 0L, length(native$fitted) == length(values), gain > -1e-6)
  }
}
