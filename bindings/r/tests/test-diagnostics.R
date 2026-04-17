assert_true <- function(condition, message) {
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

repo_root <- normalizePath(file.path(getwd(), "innovate"))
bindings_root <- file.path(repo_root, "bindings", "r")

source(file.path(bindings_root, "R", "kernel_bridge.R"))

time <- c(0, 1, 2, 3, 4)
observed <- c(0.02, 0.06, 0.12, 0.25, 0.41)

fit <- kernel_fit_model(
  kernel_request(
    operation = "fit_model",
    model_key = "bass",
    payload = list(
      inputs = list(time = time, observed = observed),
      model_kwargs = list()
    )
  )
)

diagnostics <- kernel_extract_diagnostics(fit)
assert_true(is.list(diagnostics), "Diagnostics helper should return a list")
assert_true(
  "support_level" %in% names(diagnostics),
  "Diagnostics helper should expose the support level"
)
assert_true(
  "uncertainty" %in% names(diagnostics),
  "Diagnostics helper should expose the uncertainty summary"
)

diagnose <- kernel_diagnose_model(
  kernel_request(
    operation = "diagnose_model",
    model_key = "bass",
    payload = list(
      inputs = list(time = time, observed = observed),
      state = fit$state
    )
  )
)

assert_true("diagnostics" %in% names(diagnose), "Diagnosis should include diagnostics")
assert_true(
  diagnose$diagnostics$support_level == "supported",
  "Diagnosis should report supported diagnostics"
)
assert_true(
  diagnose$diagnostics$uncertainty$report_type == "point_estimate",
  "Diagnosis should preserve the uncertainty summary"
)
