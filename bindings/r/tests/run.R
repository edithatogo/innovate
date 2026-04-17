repo_root <- normalizePath(file.path(getwd(), "innovate"))

source(file.path(repo_root, "bindings", "r", "tests", "test-architecture.R"))
source(file.path(repo_root, "bindings", "r", "tests", "test-contract.R"))

cat("R binding tests completed\n")
