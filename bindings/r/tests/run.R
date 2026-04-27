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

repo_root <- find_repo_root()
setwd(repo_root)

source(file.path(repo_root, "bindings", "r", "tests", "test-architecture.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-contract.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-wrappers.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-diagnostics.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-packaging.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-documentation.R"))

cat("R binding tests completed\n")
