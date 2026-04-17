assert_true <- function(condition, message) {
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

repo_root <- normalizePath(file.path(getwd(), "innovate"))
bindings_root <- file.path(repo_root, "bindings", "r")

source(file.path(bindings_root, "R", "kernel_bridge.R"))

example_script <- file.path(bindings_root, "examples", "end_to_end.R")

assert_true(
  file.exists(example_script),
  "Expected an installable end-to-end example script"
)

source(example_script)

stable_schema_version <- unique(kernel_discover_models()$schema_version)
assert_true(
  identical(kernel_schema_version(), stable_schema_version[[1]]),
  "Kernel schema version should match discovery results"
)
