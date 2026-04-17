assert_true <- function(condition, message) {
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

repo_root <- normalizePath(file.path(getwd(), "innovate"))
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
    model_kwargs = list(p0 = c(0.05, 0.3, 0.5))
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
  identical(response$schema_version, "1.0"),
  "Kernel discovery should return the stable schema version"
)
assert_true(
  is.list(response$models),
  "Kernel discovery should return a model list"
)
assert_true(
  length(response$models) > 0,
  "Kernel discovery should expose at least one stable model"
)
