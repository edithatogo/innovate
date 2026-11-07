# Thin native binding. Scientific computation and validation belong to Rust.
native_library <- function() {
  supplied <- Sys.getenv("INNOVATE_NATIVE_LIBRARY", "")
  if (nzchar(supplied)) return(normalizePath(supplied, mustWork = TRUE))
  suffix <- if (Sys.info()[["sysname"]] == "Darwin") ".dylib" else .Platform$dynlib.ext
  found <- system.file("lib", paste0("libinnovate_ffi", suffix), package = "innovate.R")
  if (!nzchar(found)) stop("Install the native library or set INNOVATE_NATIVE_LIBRARY", call. = FALSE)
  found
}

kernel_call <- function(operation, model_key = NULL, payload = list()) {
  if (!length(payload)) payload <- structure(list(), names = character())
  # jsonlite otherwise auto-unboxes a length-one numeric vector into a scalar.
  # Keep schema array fields as arrays while parameter/options scalars stay scalars.
  array_fields <- c("time", "t", "observed", "y", "values", "adoption", "share", "launch_times", "initial_values", "initial_agents")
  preserve_arrays <- function(value) {
    if (!is.list(value)) return(value)
    fields <- names(value)
    for (i in seq_along(value)) {
      if (!is.null(fields) && fields[[i]] == "covariates" && is.list(value[[i]]) &&
          !is.null(names(value[[i]]))) {
        value[[i]] <- lapply(value[[i]], function(x) if (is.atomic(x) && !is.null(x)) I(x) else x)
      } else if (!is.null(fields) && fields[[i]] %in% array_fields &&
          is.atomic(value[[i]]) && !is.null(value[[i]]) && !is.matrix(value[[i]])) {
        value[[i]] <- I(value[[i]])
      } else {
        value[i] <- list(preserve_arrays(value[[i]]))
      }
    }
    value
  }
  payload <- preserve_arrays(payload)
  request <- jsonlite::toJSON(list(schema_version = "2.0", operation = operation,
    model_key = model_key, payload = payload), auto_unbox = TRUE, null = "null", digits = NA)
  response <- jsonlite::fromJSON(.Call(innovate_call_R, native_library(), request), simplifyVector = FALSE)
  if (!is.null(response$error)) stop(sprintf("%s: %s", response$error$code, response$error$message), call. = FALSE)
  response
}

kernel_discover_models <- function() kernel_call("discover_models")$result
kernel_fit_model <- function(model_key, payload) kernel_call("fit_model", model_key, payload)
kernel_predict_model <- function(model_key, payload) kernel_call("predict_model", model_key, payload)
kernel_simulate_model <- function(model_key, payload) kernel_call("simulate_model", model_key, payload)
kernel_summarize_model <- function(model_key, payload) kernel_call("summarize_model", model_key, payload)
kernel_diagnose_model <- function(model_key, payload) kernel_call("diagnose_model", model_key, payload)
