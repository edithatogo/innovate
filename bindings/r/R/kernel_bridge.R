KERNEL_SCHEMA_VERSION <- "1.0"

kernel_repo_root <- function() {
  for (start in normalizePath(c(getwd(), file.path(getwd(), "innovate")), mustWork = FALSE)) {
    candidate <- start
    repeat {
      if (
        dir.exists(file.path(candidate, "src")) &&
          file.exists(file.path(candidate, "bindings", "r", "DESCRIPTION"))
      ) {
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

kernel_repo_root_or_null <- function() {
  tryCatch(kernel_repo_root(), error = function(err) NULL)
}

kernel_bindings_root <- function() {
  repo_root <- kernel_repo_root_or_null()
  if (!is.null(repo_root)) {
    return(file.path(repo_root, "bindings", "r"))
  }

  installed_root <- system.file(package = "innovate.R")
  if (nzchar(installed_root)) {
    return(installed_root)
  }

  stop("Unable to locate the innovate.R package root", call. = FALSE)
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
  installed_script <- system.file("python", "kernel_bridge.py", package = "innovate.R")
  if (nzchar(installed_script)) {
    return(installed_script)
  }

  file.path(kernel_bindings_root(), "inst", "python", "kernel_bridge.py")
}

kernel_request <- function(operation, model_key = NULL, payload = list(), metadata = list(), schema_version = KERNEL_SCHEMA_VERSION) {
  if (!is.character(operation) || length(operation) != 1L || !nzchar(operation)) {
    stop("kernel_request() requires a non-empty operation", call. = FALSE)
  }

  if (operation != "discover_models" && (is.null(model_key) || !nzchar(model_key))) {
    stop(sprintf("Kernel operation '%s' requires a model_key", operation), call. = FALSE)
  }

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
  args <- if (identical(command, "uv")) {
    c("run", "python", shQuote(kernel_bridge_script()), shQuote(request_path), shQuote(response_path))
  } else {
    c(shQuote(kernel_bridge_script()), shQuote(request_path), shQuote(response_path))
  }

  repo_root <- kernel_repo_root_or_null()
  env <- character()
  if (!is.null(repo_root)) {
    src_path <- file.path(repo_root, "src")
    existing_pythonpath <- Sys.getenv("PYTHONPATH")
    pythonpath <- if (nzchar(existing_pythonpath)) {
      paste(src_path, existing_pythonpath, sep = .Platform$path.sep)
    } else {
      src_path
    }
    env <- c(paste0("PYTHONPATH=", pythonpath))
  }
  output <- system2(command, args = args, stdout = TRUE, stderr = TRUE, env = env)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop(paste(output, collapse = "\n"), call. = FALSE)
  }

  jsonlite::fromJSON(response_path, simplifyVector = FALSE)
}

kernel_error_to_condition <- function(error, response = NULL) {
  message <- if (is.list(error) && !is.null(error$message)) error$message else "Unknown kernel error"
  condition <- structure(
    list(
      message = message,
      error = error,
      response = response
    ),
    class = c("innovate_kernel_error", "error", "condition")
  )
  stop(condition)
}

kernel_table_to_r <- function(result) {
  columns <- result$columns
  rows <- result$rows
  if (length(rows) == 0L) {
    return(stats::setNames(data.frame(matrix(ncol = length(columns), nrow = 0L)), columns))
  }

  column_values <- lapply(seq_along(columns), function(index) {
    vapply(rows, function(row) row[[index]], FUN.VALUE = rows[[1]][[index]])
  })
  names(column_values) <- columns
  as.data.frame(column_values, stringsAsFactors = FALSE, optional = TRUE)
}

kernel_array_to_r <- function(result) {
  values <- unlist(result$values, use.names = FALSE)
  shape <- as.integer(unlist(result$shape, use.names = FALSE))
  if (length(shape) == 0L) {
    return(values)
  }
  array(values, dim = shape)
}

kernel_discovery_to_r <- function(response) {
  models <- response$models
  if (length(models) == 0L) {
    return(data.frame())
  }

  data.frame(
    schema_version = rep(response$schema_version, length(models)),
    key = vapply(models, function(model) model$key, character(1L)),
    family = vapply(models, function(model) model$family, character(1L)),
    import_path = vapply(models, function(model) model$import_path, character(1L)),
    stability = vapply(models, function(model) model$stability, character(1L)),
    supports_covariates = vapply(models, function(model) isTRUE(model$supports_covariates), logical(1L)),
    supports_multivariate_output = vapply(models, function(model) isTRUE(model$supports_multivariate_output), logical(1L)),
    supported_backends = vapply(models, function(model) paste(unlist(model$supported_backends, use.names = FALSE), collapse = ", "), character(1L)),
    optional_dependencies = vapply(models, function(model) paste(unlist(model$optional_dependencies, use.names = FALSE), collapse = ", "), character(1L)),
    supports_simulation = vapply(models, function(model) isTRUE(model$supports_simulation), logical(1L)),
    supports_summarize = vapply(models, function(model) isTRUE(model$supports_summarize), logical(1L)),
    stringsAsFactors = FALSE
  )
}

kernel_value_to_r <- function(value) {
  if (is.null(value)) {
    return(NULL)
  }
  if (!is.list(value)) {
    return(value)
  }
  if (all(c("columns", "rows") %in% names(value))) {
    return(kernel_table_to_r(value))
  }
  if (all(c("shape", "dtype", "values") %in% names(value))) {
    return(kernel_array_to_r(value))
  }
  if (!is.null(value$models) && !is.null(value$schema_version) && is.null(value$operation)) {
    return(kernel_discovery_to_r(value))
  }
  if (!is.null(value$error) && !is.null(value$operation)) {
    return(kernel_response_to_r(value))
  }

  lapply(value, kernel_value_to_r)
}

kernel_response_to_r <- function(response) {
  if (!is.list(response)) {
    stop("Kernel responses must be lists", call. = FALSE)
  }

  if (!is.null(response$error)) {
    kernel_error_to_condition(response$error, response = response)
  }

  if (!is.null(response$operation) && !is.null(response$result)) {
    result <- kernel_value_to_r(response$result)
    attr(result, "kernel_operation") <- response$operation
    attr(result, "kernel_schema_version") <- response$schema_version
    attr(result, "kernel_metadata") <- response$metadata
    return(result)
  }

  if (!is.null(response$models) && !is.null(response$schema_version)) {
    return(kernel_discovery_to_r(response))
  }

  kernel_value_to_r(response)
}

kernel_schema_version <- function() {
  KERNEL_SCHEMA_VERSION
}

kernel_discover_models <- function() {
  kernel_response_to_r(kernel_call(kernel_request(operation = "discover_models")))
}

kernel_fit_model <- function(request) {
  kernel_response_to_r(kernel_call(request))
}

kernel_predict_model <- function(request) {
  kernel_response_to_r(kernel_call(request))
}

kernel_simulate_model <- function(request) {
  kernel_response_to_r(kernel_call(request))
}

kernel_summarize_model <- function(request) {
  kernel_response_to_r(kernel_call(request))
}

kernel_diagnose_model <- function(request) {
  kernel_response_to_r(kernel_call(request))
}

kernel_extract_diagnostics <- function(result) {
  if (!is.list(result)) {
    stop("Diagnostics can only be extracted from list-like kernel results", call. = FALSE)
  }

  if (!is.null(result$diagnostics)) {
    return(result$diagnostics)
  }

  diagnostics <- attr(result, "kernel_diagnostics", exact = TRUE)
  if (!is.null(diagnostics)) {
    return(diagnostics)
  }

  stop("Kernel result does not expose diagnostics", call. = FALSE)
}
