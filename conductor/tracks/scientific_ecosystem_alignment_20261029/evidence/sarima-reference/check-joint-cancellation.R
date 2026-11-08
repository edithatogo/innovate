set.seed(987)
values <- as.numeric(stats::arima.sim(
  list(ar = 0.98, ma = -0.96), n = 800L, n.start = 1000L
))
order <- c(1L, 0L, 1L)
reference <- stats::arima(values, order = order, method = "ML",
  include.mean = FALSE, transform.pars = FALSE, init = c(0.98, -0.96),
  optim.control = list(maxit = 2000L, reltol = 1e-12, ndeps = rep(1e-6, 2)))
native <- innovate.R::kernel_call("preprocess_series", payload = list(
  method = "sarima_fit", values = values, order = order, seasonal_order = integer(4)
))$result
gap <- native$log_likelihood - as.numeric(stats::logLik(reference))
error <- max(abs(unlist(native$coefficients) - stats::coef(reference)))
cat(sprintf("Likelihood gap %.9g, coefficient error %.9g\n", gap, error))
stopifnot(reference$code == 0L, abs(gap) < 1e-6, error < 3e-5)
