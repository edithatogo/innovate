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
  native <- innovate.R::kernel_seasonal_decomposition(values, 4, 13, robust)
  for (component in colnames(reference)) {
    expected <- reference[, component]
    stopifnot(max(abs(expected - fixture$reference[[mode]][[component]])) < tolerance)
    field <- if (component == "remainder") "residual" else component
    actual <- unlist(native[[field]], use.names = FALSE)
    stopifnot(length(actual) == length(expected), all(is.finite(actual)))
    error <- max(abs(actual - expected))
    cat(sprintf("%s %s %.15g\n", mode, field, error))
    if (!robust) stopifnot(error < tolerance)
  }
}
