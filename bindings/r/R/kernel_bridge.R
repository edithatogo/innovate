# Thin native binding. Scientific computation and validation belong to Rust.
native_library <- function() {
  supplied <- Sys.getenv("INNOVATE_NATIVE_LIBRARY", "")
  if (nzchar(supplied)) return(normalizePath(supplied, mustWork = TRUE))
  suffix <- if (Sys.info()[["sysname"]] == "Darwin") ".dylib" else .Platform$dynlib.ext
  if (.Platform$OS.type == "windows") {
    cargo_dll <- system.file("lib", "innovate_ffi.dll", package = "innovate.R")
    if (nzchar(cargo_dll)) return(cargo_dll)
  }
  found <- system.file("lib", paste0("libinnovate_ffi", suffix), package = "innovate.R")
  if (!nzchar(found)) stop("Install the native library or set INNOVATE_NATIVE_LIBRARY", call. = FALSE)
  found
}

kernel_call <- function(operation, model_key = NULL, payload = list(), raw = FALSE) {
  if (!length(payload)) payload <- structure(list(), names = character())
  # jsonlite otherwise auto-unboxes a length-one numeric vector into a scalar.
  # Keep schema array fields as arrays while parameter/options scalars stay scalars.
  array_fields <- c("time", "dates", "adoption_rate", "t", "observed", "y", "values", "adoption", "share", "launch_times", "initial_values", "initial_agents", "event_times", "event_effects", "event_labels", "node_labels", "covariates", "innovation_multipliers", "imitation_multipliers", "agent_service_ticks", "agent_priorities", "edge_weights", "latent_time", "latent_values", "detected_breakpoints", "groups", "components", "weights")
  if (identical(operation, "preprocess_series") && identical(payload$method, "sarima_filter")) {
    array_fields <- c(array_fields, "coefficients")
  }
  if (identical(operation, "summarize_causal")) {
    array_fields <- c(array_fields, "pre_treated", "post_treated", "pre_control",
      "post_control", "actual", "counterfactual", "periods", "coefficients",
      "standard_errors", "group_labels", "effects", "gamma_range",
      "treated_outcomes", "control_outcomes", "supporting_evidence", "caveats",
      "assumptions", "limitations", "data_sources", "outcomes", "treatment", "spillover_regions",
      "confounders", "effect_modifiers")
  }
  if (identical(operation, "summarize_causal") && identical(payload$method, "policy_scenario")) {
    array_fields <- c(array_fields, "baseline", "intervention", "assumptions")
  }
  if (operation %in% c("validate_dataset", "ingest_dataset")) {
    array_fields <- c(array_fields, "denominator", "value", "unit_id", "product_id",
      "target", "weight", "product_labels", "transform_steps")
  }
  if (identical(operation, "update_streaming_forecast")) {
    array_fields <- c(array_fields, "previous_time", "previous_observed",
      "new_time", "new_observed", "assumptions")
  }
  if (identical(operation, "compose_regime_ensemble")) {
    array_fields <- c(array_fields, "assumptions")
  }
  if (identical(operation, "summarize_competition_equilibrium")) {
    array_fields <- c(array_fields, "market_potentials")
  }
  if (operation %in% c("residual_diagnostic_series", "render_residual_diagnostics_svg")) {
    array_fields <- c(array_fields, "residuals")
  }
  preserve_arrays <- function(value) {
    if (!is.list(value)) return(value)
    fields <- names(value)
    for (i in seq_along(value)) {
      # Event times and named overrides are scalars, unlike observation times.
      if (!is.null(fields) && fields[[i]] == "parameter_events") next
      if (!is.null(fields) && (fields[[i]] %in% c("covariates", "covariate_overrides", "effect_modifiers", "observed", "y", "values") ||
          (identical(operation, "compose_regime_ensemble") && fields[[i]] == "predictions") ||
          (identical(operation, "summarize_causal") && identical(payload$method, "policy_specification") &&
           fields[[i]] == "data")) && is.list(value[[i]]) &&
          !is.null(names(value[[i]]))) {
        value[[i]] <- lapply(value[[i]], function(x) if (is.atomic(x) && !is.null(x)) I(x) else x)
      } else if (!is.null(fields) && (fields[[i]] %in% array_fields ||
          (fields[[i]] == "source" && identical(value$kind, "network_edges"))) &&
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
    model_key = model_key, payload = payload), auto_unbox = TRUE, null = "null", digits = I(17))
  kernel_call_json(request, raw = raw)
}

# Send an existing JSON envelope without re-encoding numbers or arrays.
kernel_call_json <- function(request, raw = FALSE) {
  if (!is.character(request) || length(request) != 1L || is.na(request)) {
    stop("request must be a single JSON string", call. = FALSE)
  }
  if (!is.logical(raw) || length(raw) != 1L || is.na(raw)) {
    stop("raw must be a single logical value", call. = FALSE)
  }
  library <- native_library()
  # Hold a loader reference across JSON calls for the Rust registry.
  paths <- vapply(getLoadedDLLs(), function(info) info[["path"]], character(1))
  if (!library %in% paths) dyn.load(library, local = TRUE, now = TRUE)
  decode_kernel_response(.Call(innovate_call_R, library, request), raw)
}

kernel_scenario_grid_ipc <- function(data, payload) {
  request <- jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null", digits = I(17))
  decode_kernel_response(.Call(innovate_scenario_grid_ipc_R, native_library(), request, data))
}

kernel_scenario_executions_ipc <- function(response) {
  scenario_output(response, innovate_scenario_executions_ipc_R)
}

kernel_scenario_comparisons_ipc <- function(response) {
  scenario_output(response, innovate_scenario_comparisons_ipc_R)
}

kernel_paired_outcomes_ipc <- function(payload) {
  scenario_output(payload, innovate_paired_outcomes_ipc_R)
}

scenario_output <- function(response, symbol) {
  if (is.list(response)) {
    response <- jsonlite::toJSON(response, auto_unbox = TRUE, null = "null", digits = I(17))
  }
  output <- .Call(symbol, native_library(), response)
  if (is.raw(output)) output else decode_kernel_response(output)
}

decode_kernel_response <- function(response_json, raw = FALSE) {
  response <- jsonlite::fromJSON(response_json, simplifyVector = FALSE)
  if (!is.null(response$error)) {
    details <- response$error
    message <- sprintf("%s: %s", details$code, details$message)
    if (!is.null(details$validation)) {
      stop(errorCondition(message, class = "innovate_dataset_validation_error",
        code = details$code, validation = details$validation))
    }
    stop(message, call. = FALSE)
  }
  if (raw) response_json else response
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
  if (inherits(dates, "POSIXt")) {
    dates <- format(dates, "%Y-%m-%dT%H:%M:%OS6Z", tz = "UTC")
  }
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

kernel_sarima_fit <- function(values, order, seasonal_order = c(0L, 0L, 0L, 0L), historical = FALSE) {
  fitted <- kernel_call("preprocess_series", payload = list(method = "sarima_fit",
    values = values, order = order, seasonal_order = seasonal_order, historical = historical))$result$fitted
  vapply(fitted, function(value) if (is.null(value)) NA_real_ else value, numeric(1))
}

export_ipc_file <- function(data) {
  if (!is.raw(data)) stop("data must be raw IPC bytes", call. = FALSE)
  scenario_output(data, innovate_export_ipc_file_R)
}

export_ipc_stream <- function(data) {
  if (!is.raw(data)) stop("data must be raw IPC bytes", call. = FALSE)
  scenario_output(data, innovate_export_ipc_stream_R)
}

invoke_registered_extension <- function(callback, selection, payload) {
  resolved <- kernel_call("resolve_extension_manifest", payload = selection)$result
  output <- callback(resolved, payload)
  if (identical(resolved$extension_point, "dataset_provider")) {
    return(kernel_call("ingest_dataset", payload = list(dataset = output))$result)
  }
  if (identical(resolved$extension_point, "model_registry")) {
    return(kernel_call("resolve_extension_model_aliases", payload = list(aliases = output))$result)
  }
  if (identical(resolved$extension_point, "diagnostics")) {
    return(kernel_call("summarize_extension_diagnostics", payload = list(checks = output))$result)
  }
  kernel_call("import_extension_artifact", payload = list(artifact = output))$result
}
