library(innovate.R)
report <- jsonlite::fromJSON(commandArgs(trailingOnly=TRUE)[[1]], simplifyVector=FALSE)
values <- function(x) as.numeric(unlist(x, use.names=FALSE))
binary64 <- function(tokens) {
  offsets <- seq.int(1L, 15L, 2L)
  vapply(tokens, function(token) {
    bytes <- as.raw(strtoi(substring(token, offsets, offsets + 1L), base=16L))
    readBin(bytes, "double", n=1L, size=8L, endian="big")
  }, numeric(1))
}
for (case in report$valid) {
  time <- binary64(case$time_bits)
  rates <- binary64(case$rate_bits)
  result <- tryCatch(kernel_categorize_adopters(time, rates),
    error=function(e) stop(case$name, ": ", conditionMessage(e), call.=FALSE))
  if (!identical(values(result$time), time)) stop(case$name, " changed time values")
  if (!identical(values(result$adoption_rate), rates)) stop(case$name, " changed weights")
  if (!identical(unlist(result$category, use.names=FALSE),
                 unlist(case$categories, use.names=FALSE))) stop(case$name, " changed the labels")
}
for (case in report$invalid) {
  result <- tryCatch({
    kernel_categorize_adopters(values(case$time), values(case$rates))
    "Successful response"
  }, error=conditionMessage)
  if (!grepl(case$error, result, fixed=TRUE)) stop(case$name, ": unexpected result ", result)
}

case <- report$logistic_fit
fit <- kernel_fit_model("logistic",
  list(inputs=list(time=values(case$time), observed=values(case$observed)),
       fit_options=list(initial_parameters=case$initial)))$result
stopifnot(all(abs(values(fit$parameters[c("L", "k", "x0")]) - c(100, 1, 10)) <= 1e-8))
classified <- kernel_categorize_fitted_adopters("logistic",
  list(inputs=list(time=values(case$prediction_time)), state=fit$state))
stopifnot(all(abs(values(classified$adoption_rate) - values(case$analytic_rates)) <= 1e-8),
          identical(unlist(classified$category), unlist(case$categories)))
cat(R.version.string, ": ", length(report$valid), " reference cases, ",
    length(report$invalid), " error cases and the fitted Logistic case passed\n", sep="")
