# Generate a stationary multiplicative SARIMA reference with fixed coefficients.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
i <- seq_len(40)
values <- sin(i * 0.71) + cos(i * 0.23) / 2
order <- c(1L, 0L, 1L)
seasonal_order <- c(1L, 0L, 1L, 4L)
coefficients <- c(ar1 = 0.35, ma1 = -0.20, sar1 = 0.25, sma1 = 0.10)
fit <- stats::arima(
  values, order = order,
  seasonal = list(order = seasonal_order[1:3], period = seasonal_order[4]),
  include.mean = FALSE, fixed = unname(coefficients), method = "ML"
)
residual <- as.numeric(stats::residuals(fit))
stopifnot(length(residual) == length(values), all(is.finite(residual)))
jsonlite::write_json(
  list(runtime = as.character(getRversion()), values = values, order = order,
       seasonal_order = seasonal_order, coefficients = unname(coefficients),
       fitted = values - residual, residual = residual),
  args[[1L]], digits = NA, pretty = TRUE, auto_unbox = TRUE
)
