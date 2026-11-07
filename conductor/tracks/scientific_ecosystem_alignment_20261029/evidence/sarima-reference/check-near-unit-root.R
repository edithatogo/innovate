# Compare installed Rust binding fits against independent R maximum likelihood.
for (term in c("ar", "ma")) {
  set.seed(1407)
  order <- if (term == "ar") c(1L, 0L, 0L) else c(0L, 0L, 1L)
  for (coefficient in c(0.985, -0.985)) {
    model <- setNames(list(coefficient), term)
    values <- as.numeric(stats::arima.sim(model, n = 240L, n.start = if (term == "ar") 800L else 1L))
    reference <- stats::arima(values, order = order, method = "ML", include.mean = FALSE)
    native <- innovate.R::kernel_call("preprocess_series", payload = list(
      method = "sarima_fit", values = values, order = order, seasonal_order = integer(4)
    ))$result
    estimate <- unlist(native$coefficients)
    gain <- native$log_likelihood - as.numeric(stats::logLik(reference))
    cat(sprintf("%s %+.3f: R %.9f, Rust %.9f, Rust minus R likelihood %.9g\n",
      term, coefficient, stats::coef(reference), estimate, gain))
    stopifnot(reference$code == 0L, length(native$fitted) == length(values),
      length(estimate) == 1L, abs(estimate) < 1,
      abs(estimate - stats::coef(reference)) < 2e-5, abs(gain) < 1e-6)
  }
}
