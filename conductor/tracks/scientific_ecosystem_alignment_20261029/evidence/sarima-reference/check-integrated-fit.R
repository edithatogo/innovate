set.seed(42)
values <- diffinv(diffinv(as.numeric(stats::arima.sim(list(ar = 0.45), n = 240)), lag = 4))
order <- c(1L, 1L, 0L)
seasonal_order <- c(0L, 1L, 0L, 4L)
reference <- function(series, fixed = NA_real_) {
  stats::arima(series, order = order, seasonal = list(order = seasonal_order[1:3], period = 4),
               method = "ML", fixed = fixed)
}
run <- function(method, ...) {
  innovate.R::kernel_call("preprocess_series", payload = c(list(
    method = method, values = values, order = order, seasonal_order = seasonal_order
  ), list(...)))$result
}
fit <- reference(values)
native <- run("sarima_fit")
fixed <- run("sarima_filter", coefficients = 0.45)
indices <- 7:40
predictions <- vapply(indices, function(i) {
  as.numeric(stats::predict(reference(values[seq_len(i - 1L)], 0.45), n.ahead = 1)$pred)
}, numeric(1))
errors <- c(
  coefficient = abs(native$coefficients[[1]] - stats::coef(fit)[[1]]),
  likelihood = abs(native$log_likelihood - as.numeric(stats::logLik(fit))),
  prediction = max(abs(unlist(fixed$fitted[indices]) - predictions))
)
print(errors)
stopifnot(fit$code == 0L, length(fixed$fitted) == length(values),
          all(vapply(fixed$fitted[1:5], is.null, logical(1))),
          all(errors < c(5e-6, 1e-6, 1e-8)))
