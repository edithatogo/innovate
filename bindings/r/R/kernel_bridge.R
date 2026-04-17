KERNEL_SCHEMA_VERSION <- "1.0"

kernel_repo_root <- function() {
  candidates <- c(
    file.path(getwd(), "innovate"),
    getwd()
  )

  for (candidate in candidates) {
    if (dir.exists(file.path(candidate, "src"))) {
      return(normalizePath(candidate))
    }
  }

  stop("Unable to locate the innovate repository root", call. = FALSE)
}

kernel_bindings_root <- function() {
  file.path(kernel_repo_root(), "bindings", "r")
}

kernel_python_command <- function() {
  command <- Sys.getenv("INNOVATE_PYTHON_COMMAND", "uv")
  if (nzchar(Sys.which(command))) {
    return(command)
  }
  if (command != "python3" && nzchar(Sys.which("python3"))) {
    return("python3")
  }
  stop("Unable to locate a Python launcher for the kernel bridge", call. = FALSE)
}

kernel_bridge_script <- function() {
  file.path(kernel_bindings_root(), "inst", "python", "kernel_bridge.py")
}

kernel_request <- function(operation, model_key = NULL, payload = list(), metadata = list(), schema_version = KERNEL_SCHEMA_VERSION) {
  list(
    schema_version = schema_version,
    operation = operation,
    model_key = model_key,
    payload = payload,
    metadata = metadata
  )
}

kernel_call <- function(request) {
  request_path <- tempfile("innovate-kernel-request-", fileext = ".json")
  response_path <- tempfile("innovate-kernel-response-", fileext = ".json")
  old_wd <- getwd()
  on.exit({
    setwd(old_wd)
    if (file.exists(request_path)) {
      unlink(request_path)
    }
    if (file.exists(response_path)) {
      unlink(response_path)
    }
  }, add = TRUE)

  jsonlite::write_json(request, request_path, auto_unbox = TRUE, null = "null", digits = NA, pretty = TRUE)

  command <- kernel_python_command()
  args <- c(
    "run",
    "python",
    kernel_bridge_script(),
    request_path,
    response_path
  )
  env <- c(
    paste0("PYTHONPATH=", file.path(kernel_repo_root(), "src"))
  )
  setwd(kernel_repo_root())
  output <- system2(command, args = args, stdout = TRUE, stderr = TRUE, env = env)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop(paste(output, collapse = "\n"), call. = FALSE)
  }

  jsonlite::fromJSON(response_path, simplifyVector = FALSE)
}

kernel_schema_version <- function() {
  KERNEL_SCHEMA_VERSION
}

kernel_discover_models <- function() {
  kernel_call(kernel_request(operation = "discover_models"))
}

kernel_fit_model <- function(request) {
  kernel_call(request)
}

kernel_predict_model <- function(request) {
  kernel_call(request)
}

kernel_simulate_model <- function(request) {
  kernel_call(request)
}

kernel_summarize_model <- function(request) {
  kernel_call(request)
}

kernel_diagnose_model <- function(request) {
  kernel_call(request)
}
