find_repo_root <- function() {
  args <- commandArgs(FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script_path <- if (length(file_arg) > 0L) {
    sub("^--file=", "", file_arg[[1]])
  } else {
    positional <- args[!grepl("^--", args)]
    positional <- positional[file.exists(positional)]
    if (length(positional) > 0L) positional[[length(positional)]] else ""
  }

  candidates <- normalizePath(
    c(getwd(), file.path(getwd(), "innovate"), if (nzchar(script_path)) dirname(script_path) else character()),
    mustWork = FALSE
  )

  for (start in candidates) {
    candidate <- start
    repeat {
      if (file.exists(file.path(candidate, "bindings", "r", "DESCRIPTION"))) {
        return(candidate)
      }

      parent <- dirname(candidate)
      if (identical(parent, candidate)) {
        break
      }
      candidate <- parent
    }
  }

  stop("Unable to locate the innovate repository root", call. = FALSE)
}

assert_true <- function(condition, message) {
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

run_integration_tests <- identical(Sys.getenv("INNOVATE_RUN_INTEGRATION_TESTS"), "true")
repo_root <- NULL

if (run_integration_tests) {
  repo_root <- find_repo_root()
  setwd(repo_root)
  source(file.path(repo_root, "bindings", "r", "R", "kernel_bridge.R"))
} else if (requireNamespace("innovate.R", quietly = TRUE)) {
  library(innovate.R)
} else {
  repo_root <- find_repo_root()
  setwd(repo_root)
  source(file.path(repo_root, "bindings", "r", "R", "kernel_bridge.R"))
}

assert_true(
  identical(kernel_schema_version(), "1.0"),
  "Kernel schema version should be stable"
)

request <- kernel_request(
  operation = "fit_model",
  model_key = "bass",
  payload = list(inputs = list(time = c(0, 1), observed = c(0.1, 0.2)))
)
assert_true(
  identical(request$schema_version, "1.0"),
  "Kernel requests should preserve the schema version"
)
assert_true(
  identical(request$operation, "fit_model"),
  "Kernel requests should preserve the operation"
)
assert_true(
  identical(request$model_key, "bass"),
  "Kernel requests should preserve the model key"
)

converted <- kernel_response_to_r(list(
  schema_version = "1.0",
  operation = "predict_model",
  result = list(shape = list(2L), dtype = "float64", values = list(0.1, 0.2)),
  error = NULL,
  metadata = list(source = "test")
))
assert_true(
  identical(as.numeric(converted), c(0.1, 0.2)),
  "Kernel array responses should convert to R arrays"
)
assert_true(
  identical(attr(converted, "kernel_operation"), "predict_model"),
  "Converted responses should preserve kernel metadata"
)

captured_error <- tryCatch(
  {
    kernel_response_to_r(list(
      schema_version = "1.0",
      operation = "fit_model",
      result = NULL,
      error = list(code = "invalid_request", message = "boom"),
      metadata = list()
    ))
    NULL
  },
  error = function(err) err
)
assert_true(
  inherits(captured_error, "innovate_kernel_error"),
  "Kernel errors should map to a dedicated R condition class"
)

if (run_integration_tests) {
  source(file.path(repo_root, "bindings", "r", "tests", "test-architecture.R"))
  source(file.path(repo_root, "bindings", "r", "tests", "test-contract.R"))
  source(file.path(repo_root, "bindings", "r", "tests", "test-wrappers.R"))
  source(file.path(repo_root, "bindings", "r", "tests", "test-diagnostics.R"))
  source(file.path(repo_root, "bindings", "r", "tests", "test-packaging.R"))
  source(file.path(repo_root, "bindings", "r", "tests", "test-documentation.R"))
}

cat("R binding tests completed\n")
