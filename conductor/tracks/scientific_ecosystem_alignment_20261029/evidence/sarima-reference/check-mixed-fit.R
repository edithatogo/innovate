# Compare the Rust estimator with an independent maximum likelihood fit.
set.seed(42)
values <- as.numeric(stats::arima.sim(
  list(ar = c(0.45, 0, 0, 0.35, -0.1575), ma = c(-0.2, 0, 0, 0.3, -0.06)),
  n = 240
))
order <- c(1L, 0L, 1L)
seasonal_order <- c(1L, 0L, 1L, 4L)
reference <- stats::arima(
  values, order = order,
  seasonal = list(order = seasonal_order[1:3], period = seasonal_order[4]),
  method = "ML", include.mean = FALSE
)
native <- innovate.R::kernel_call("preprocess_series", payload = list(
  method = "sarima_fit", values = values, order = order, seasonal_order = seasonal_order
))$result
stopifnot(reference$code == 0L, length(native$fitted) == length(values),
          length(native$coefficients) == length(stats::coef(reference)))
coefficient_error <- max(abs(unlist(native$coefficients) - stats::coef(reference)))
likelihood_error <- abs(native$log_likelihood - as.numeric(stats::logLik(reference)))
cat(sprintf("maximum coefficient error %.12g\nlog likelihood error %.12g\n",
            coefficient_error, likelihood_error))
stopifnot(coefficient_error < 5e-4, likelihood_error < 1e-6)
