assert_true <- function(condition, message) {
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

repo_root <- normalizePath(file.path(getwd(), "innovate"))
bindings_root <- file.path(repo_root, "bindings", "r")
readme_path <- file.path(bindings_root, "README.md")

readme <- paste(readLines(readme_path, warn = FALSE), collapse = "\n")

assert_true(
  grepl("## API surface", readme, fixed = TRUE),
  "Expected API surface guidance in the README"
)
assert_true(
  grepl("## Backend expectations", readme, fixed = TRUE),
  "Expected backend expectations guidance in the README"
)
assert_true(
  grepl("kernel_extract_diagnostics", readme, fixed = TRUE),
  "Expected the README to mention the diagnostics helper"
)
