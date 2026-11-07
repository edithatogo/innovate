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
  array_fields <- c("time", "dates", "adoption_rate", "t", "observed", "y", "values", "adoption", "share", "launch_times", "initial_values", "initial_agents", "event_times", "event_effects", "event_labels", "node_labels", "covariates", "innovation_multipliers", "imitation_multipliers", "agent_service_ticks", "agent_priorities", "edge_weights", "latent_time", "latent_values", "detected_breakpoints", "groups", "components", "weights")
  if (identical(operation, "preprocess_series") && identical(payload$method, "sarima_filter")) {
    array_fields <- c(array_fields, "coefficients")
  }
  preserve_arrays <- function(value) {
    if (!is.list(value)) return(value)
    fields <- names(value)
    for (i in seq_along(value)) {
      # Event times and named overrides are scalars, unlike observation times.
      if (!is.null(fields) && fields[[i]] == "parameter_events") next
      if (!is.null(fields) && fields[[i]] %in% c("covariates", "observed", "y", "values") && is.list(value[[i]]) &&
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

kernel_fit_batch <- function(model_key, datasets, workers = 1L) kernel_call("fit_batch", model_key, list(datasets = datasets, workers = workers))$result$datasets
kernel_predict_batch <- function(model_key, datasets, workers = 1L) kernel_call("predict_batch", model_key, list(datasets = datasets, workers = workers))$result$datasets

kernel_categorize_adopters <- function(time, rates) {
  kernel_call("categorize_adopters", payload = list(time = time, adoption_rate = rates))$result
}

kernel_categorize_fitted_adopters <- function(model_key, payload) {
  rates <- kernel_predict_model(model_key, payload)$result$rates
  if (is.null(rates) || !all(lengths(rates) == 1L)) {
    stop("adopter classification requires one native rate per time", call. = FALSE)
  }
  time <- payload$inputs$time
  if (is.null(time)) time <- payload$inputs$t
  kernel_categorize_adopters(time, unlist(rates, use.names = FALSE))
}

kernel_cumulative_sum <- function(values) {
  kernel_call("preprocess_series", payload = list(method = "cumulative_sum", values = values))$result$values
}

kernel_rolling_average <- function(values, window) {
  kernel_call("preprocess_series", payload = list(method = "rolling_average", values = values, window = window))$result$values
}

kernel_aggregate_dates <- function(dates, values, frequency) {
  kernel_call("aggregate_dates", payload = list(dates = as.character(dates),
    values = values, frequency = frequency))$result
}

kernel_seasonal_decomposition <- function(values, period, trend_span = NULL, robust = FALSE, historical = FALSE,
                                          inner_iterations = 5L, outer_iterations = if (isTRUE(robust)) 5L else 0L,
                                          seasonal_span = NULL, low_pass_span = NULL,
                                          seasonal_degree = 1L, seasonal_jump = 1L,
                                          trend_degree = 1L, trend_jump = 1L,
                                          low_pass_degree = 1L, low_pass_jump = 1L) {
  kernel_call("preprocess_series", payload = list(method = "seasonal_decomposition",
    values = values, period = period, window = trend_span, robust = robust, historical = historical,
    inner_iterations = inner_iterations, outer_iterations = outer_iterations,
    seasonal_span = seasonal_span, low_pass_span = low_pass_span,
    seasonal_degree = seasonal_degree, seasonal_jump = seasonal_jump,
    trend_degree = trend_degree, trend_jump = trend_jump,
    low_pass_degree = low_pass_degree, low_pass_jump = low_pass_jump))$result
}

kernel_sarima_fit <- function(values, order, seasonal_order = c(0L, 0L, 0L, 0L)) {
  fitted <- kernel_call("preprocess_series", payload = list(method = "sarima_fit",
    values = values, order = order, seasonal_order = seasonal_order))$result$fitted
  vapply(fitted, function(value) if (is.null(value)) NA_real_ else value, numeric(1))
}
