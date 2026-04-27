assert_true <- function(condition, message) {
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

repo_root <- if (exists("find_repo_root")) {
  find_repo_root()
} else {
  candidate <- normalizePath(getwd(), mustWork = TRUE)
  repeat {
    if (file.exists(file.path(candidate, "bindings", "r", "DESCRIPTION"))) {
      break
    }
    parent <- dirname(candidate)
    if (identical(parent, candidate)) {
      stop("Unable to locate the innovate repository root", call. = FALSE)
    }
    candidate <- parent
  }
  candidate
}
bindings_root <- file.path(repo_root, "bindings", "r")

source(file.path(bindings_root, "R", "kernel_bridge.R"))

time <- c(0, 1, 2, 3, 4)
observed <- c(0.02, 0.06, 0.12, 0.25, 0.41)

discovery <- kernel_discover_models()
assert_true(is.data.frame(discovery), "Discovery should return a data frame")
for (column in c("key", "family", "stability", "supports_simulation")) {
  assert_true(column %in% names(discovery), paste("Missing discovery column:", column))
}
assert_true("schema_version" %in% names(discovery), "Discovery should expose the schema version column")
assert_true(
  all(discovery$schema_version == "1.0"),
  "Kernel discovery should return the stable schema version"
)
assert_true(nrow(discovery) > 0, "Discovery should expose at least one stable model")

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
assert_true(!"operation" %in% names(fit), "Fit should return the converted model payload, not the raw envelope")
assert_true("state" %in% names(fit), "Fit should expose the fitted state")
assert_true("predictions" %in% names(fit), "Fit should expose fitted predictions")
assert_true(is.list(fit$state), "Fit state should be a list")

predict <- kernel_predict_model(
  kernel_request(
    operation = "predict_model",
    model_key = "bass",
    payload = list(
      inputs = list(time = time),
      state = fit$state
    )
  )
)
assert_true(
  is.numeric(predict) || is.matrix(predict) || is.data.frame(predict),
  "Predict should return an R-native numeric or tabular object"
)

summary_result <- kernel_summarize_model(
  kernel_request(
    operation = "summarize_model",
    model_key = "bass",
    payload = list(
      inputs = list(time = time, observed = observed),
      state = fit$state
    )
  )
)
assert_true("diagnostics" %in% names(summary_result), "Summaries should surface diagnostics")
assert_true("state" %in% names(summary_result), "Summaries should preserve the model state")

error_response <- list(
  schema_version = "1.0",
  operation = "fit_model",
  result = NULL,
  error = list(
    code = "invalid_request",
    message = "boom",
    operation = "fit_model",
    details = list(),
    retryable = FALSE
  ),
  metadata = list()
)

captured_error <- tryCatch(
  {
    kernel_response_to_r(error_response)
    NULL
  },
  error = function(err) {
    err
  }
)

assert_true(inherits(captured_error, "innovate_kernel_error"), "Kernel errors should map to a dedicated R condition class")
assert_true(grepl("boom", conditionMessage(captured_error)), "Kernel error message should be preserved")
