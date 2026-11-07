# Run with the installed innovate.R package and the rebuilt native library.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
fixture <- jsonlite::fromJSON(args[[1L]])
i <- 0:79
values <- 10 + 0.5 * i + rep(c(-3, 2, 1, 0), 20) + 0.2 * sin(i * 0.37)
values[41] <- values[41] + 100
tolerance <- 1e-10
stopifnot(max(abs(values - fixture$values)) < tolerance)

for (mode in c("nonrobust", "robust")) {
  robust <- mode == "robust"
  reference <- stats::stl(
    stats::ts(values, frequency = 4), s.window = 7, t.window = 13, l.window = 5,
    s.degree = 1, t.degree = 1, l.degree = 1, s.jump = 1, t.jump = 1, l.jump = 1,
    inner = 5, outer = if (robust) 5 else 0, robust = robust
  )$time.series
  native <- innovate.R::kernel_seasonal_decomposition(values, 4, 13, robust, historical = robust)
  for (component in colnames(reference)) {
    expected <- reference[, component]
    stopifnot(max(abs(expected - fixture$reference[[mode]][[component]])) < tolerance)
    field <- if (component == "remainder") "residual" else component
    actual <- unlist(native[[field]], use.names = FALSE)
    stopifnot(length(actual) == length(expected), all(is.finite(actual)))
    error <- max(abs(actual - expected))
    cat(sprintf("%s %s %.15g\n", mode, field, error))
    stopifnot(error < if (robust) 1e-9 else tolerance)
  }
}

# Inspect sensitivity of the independent result to the outer iteration count.
cat("reference outer_iterations spike_residual spike_weight zero_weights\n")
for (outer in 0:6) {
  fit <- stats::stl(
    stats::ts(values, frequency = 4), s.window = 7, t.window = 13, l.window = 5,
    s.degree = 1, t.degree = 1, l.degree = 1, s.jump = 1, t.jump = 1, l.jump = 1,
    inner = 5, outer = outer, robust = TRUE
  )
  cat(sprintf("%d %.15g %.15g %d\n", outer, fit$time.series[41, "remainder"],
              fit$weights[41], sum(fit$weights == 0)))
}

cat("partial_cycle length period max_component_error\n")
for (n in c(13L, 37L, 79L, 81L)) {
  for (period in c(3L, 4L)) {
    i <- seq_len(n)
    values <- 5 + i / 4 + sin(2 * pi * i / period) + cos(i * 0.31) / 4
    reference <- stats::stl(
      stats::ts(values, frequency = period), s.window = 7, t.window = 13,
      l.window = 5, s.degree = 1, t.degree = 1, l.degree = 1,
      s.jump = 2, t.jump = 3, l.jump = 2, inner = 2, outer = 0
    )$time.series
    native <- innovate.R::kernel_seasonal_decomposition(
      values, period, 13L, historical = TRUE, inner_iterations = 2L,
      seasonal_span = 7L, low_pass_span = 5L, seasonal_jump = 2L,
      trend_jump = 3L, low_pass_jump = 2L
    )
    actual <- cbind(unlist(native$seasonal), unlist(native$trend), unlist(native$residual))
    stopifnot(identical(dim(actual), dim(reference)))
    error <- max(abs(actual - reference))
    stopifnot(error < tolerance)
    cat(sprintf("%d %d %.15g\n", n, period, error))
  }
}
