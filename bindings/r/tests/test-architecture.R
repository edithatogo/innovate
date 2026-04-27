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

assert_true(
  file.exists(file.path(bindings_root, "DESCRIPTION")),
  "Expected R package metadata at bindings/r/DESCRIPTION"
)
assert_true(
  file.exists(file.path(bindings_root, "NAMESPACE")),
  "Expected R package namespace at bindings/r/NAMESPACE"
)
assert_true(
  file.exists(file.path(bindings_root, "R", "kernel_bridge.R")),
  "Expected R kernel bridge implementation at bindings/r/R/kernel_bridge.R"
)
assert_true(
  file.exists(file.path(bindings_root, "tests", "run.R")),
  "Expected the R test harness at bindings/r/tests/run.R"
)
