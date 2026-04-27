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

request <- kernel_request(
  operation = "fit_model",
  model_key = "bass",
  payload = list(
    inputs = list(
      time = c(0, 1, 2),
      observed = c(0.1, 0.2, 0.4)
    ),
    model_kwargs = list()
  )
)

assert_true(
  identical(request$schema_version, "1.0"),
  "Kernel requests should default to schema version 1.0"
)
assert_true(
  identical(request$operation, "fit_model"),
  "Kernel requests should preserve the requested operation"
)
assert_true(
  identical(request$model_key, "bass"),
  "Kernel requests should preserve the requested model key"
)
assert_true(
  identical(request$payload$inputs$time, c(0, 1, 2)),
  "Kernel requests should preserve numeric payload values"
)

response <- kernel_discover_models()

assert_true(
  is.data.frame(response),
  "Kernel discovery should return a data frame"
)
assert_true(
  "schema_version" %in% names(response),
  "Kernel discovery should expose the stable schema version"
)
assert_true(
  all(response$schema_version == "1.0"),
  "Kernel discovery should return the stable schema version"
)
assert_true(
  nrow(response) > 0,
  "Kernel discovery should expose at least one stable model"
)
