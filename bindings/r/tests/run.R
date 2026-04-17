repo_root <- normalizePath(file.path(getwd(), "innovate"))

source(file.path(repo_root, "bindings", "r", "tests", "test-architecture.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-contract.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-wrappers.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-diagnostics.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-packaging.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-documentation.R"))

cat("R binding tests completed\n")
