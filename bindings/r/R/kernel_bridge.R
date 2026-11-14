# Thin native binding. Scientific computation and validation belong to Rust.
# Resolve an explicit native-library override or the installed platform bundle.
# An override must exist; bundled lookup recognizes Cargo's Windows DLL name.
# No build, download or fallback interpreter is started.
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

#' Execute a schema-2.0 request in the native Rust kernel.
#'
#' Requests are encoded with jsonlite using 17 digits for numeric values. R numeric
#' precision already lost before encoding cannot be recovered; use kernel_call_json for
#' exact existing JSON. Scientific computation and validation run in Rust with no Python
#' fallback. JSON calls keep a loader reference for process-local registry state, copy
#' response text into R, and release native response memory.
#'
#' @param operation Canonical native operation name, such as "discover_models" or
#'   "predict_model".
#' @param model_key Native model key; NULL by default for operations without a selected
#'   model.
#' @param payload Operation-specific named list, defaulting to an empty JSON object. Known
#'   native array fields retain array shape even for a single element; scalar parameter
#'   fields remain scalars. NULL values are encoded as JSON null.
#' @param raw One non-missing logical value, FALSE by default. TRUE returns successful
#'   response JSON without replacing its original numeric tokens or array shapes.
#' @return A complete native response as nested lists when raw is FALSE, or the original
#'   response string when TRUE. Arrays decode as lists and JSON null as NULL. Successful
#'   envelopes may omit error; non-null native errors are raised in either mode.
#' @section Errors:
#' Native errors raise R conditions; dataset errors with validation details have class
#' innovate_dataset_validation_error and carry code and validation. JSON conversion,
#' library loading and ABI errors propagate. An invalid raw flag is rejected by
#' kernel_call_json.
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
  # Recursively protect known schema arrays from singleton auto-unboxing.
  # Preserve explicit NULL members and named scalar event/parameter overrides.
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
#' Send an existing JSON request without re-encoding its numbers or arrays.
#'
#' Use this entry point for full-width integer identifiers or externally prepared schema
#' arrays. Parsed R numeric values may round wide integers; raw=TRUE preserves their
#' response tokens. INNOVATE_NATIVE_LIBRARY overrides the installed platform-specific
#' library in the package lib directory.
#'
#' @param request Single non-missing character string containing the full schema-2.0
#'   envelope: schema_version, operation, optional model_key and payload.
#' @param raw Single non-missing logical value, FALSE by default. TRUE returns the original
#'   successful response string after native-error inspection.
#' @return The parsed native envelope as nested lists, or its exact JSON string if raw is
#'   TRUE. Even raw responses are checked for a native error before being returned.
#' @section Errors:
#' Invalid request/raw types, lengths or missing values are rejected before the native
#' call. Native errors raise R conditions; dataset errors with validation details have
#' class innovate_dataset_validation_error and carry code and validation. JSON conversion,
#' library loading and ABI errors propagate.
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

#' Execute a scenario grid on a shared Arrow IPC time column.
#'
#' The synchronous C call reads the supplied bytes; its JSON response is copied into R and
#' native allocation is released. The helper returns structured JSON, not an Arrow R table.
#'
#' @param data Raw vector containing an Arrow IPC file or stream with the required numeric
#'   time column.
#' @param payload Named list containing ordered scenarios and optional
#'   comparisons/execution controls, not a complete schema envelope. Wrap singleton numeric
#'   arrays in I() when their fields require JSON arrays; this entry point directly uses
#'   jsonlite auto-unboxing.
#' @return The decoded native response envelope as nested lists, including ordered
#'   execution rows and any requested comparisons/artifacts. Individual scenario failures
#'   remain in their rows; outer errors are raised.
#' @section Errors:
#' The native bridge rejects non-raw input, invalid IPC or grid requests. Native errors
#' raise R conditions; dataset errors with validation details have class
#' innovate_dataset_validation_error and carry code and validation. JSON conversion,
#' library loading and ABI errors propagate.
kernel_scenario_grid_ipc <- function(data, payload) {
  request <- jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null", digits = I(17))
  decode_kernel_response(.Call(innovate_scenario_grid_ipc_R, native_library(), request, data))
}

#' Encode ordered scenario execution rows as an Arrow IPC file.
#'
#' The bridge copies output bytes into R and frees native memory before returning. Passing
#' a parsed list through jsonlite can alter singleton arrays or wide numeric identifiers;
#' preserve exact JSON where those distinctions matter.
#'
#' @param response Complete scenario-grid response as a nested list or one JSON string. Use
#'   existing JSON text when full-width integer identifiers must retain their exact tokens.
#' @return A newly allocated raw vector containing a native Arrow IPC file. Rows retain
#'   their scenario execution order; nested results and errors use the native JSON columns.
#' @section Errors:
#' Malformed responses, unsupported exports, native allocation failures and invalid scalar-string inputs raise errors. Native errors raise R conditions; dataset errors with
#' validation details have class innovate_dataset_validation_error and carry code and
#' validation. JSON conversion, library loading and ABI errors propagate.
kernel_scenario_executions_ipc <- function(response) {
  scenario_output(response, innovate_scenario_executions_ipc_R)
}

#' Encode ordered scenario comparison rows as an Arrow IPC file.
#'
#' The bridge copies output bytes into R and frees native memory before returning. Passing
#' a parsed list through jsonlite can alter singleton arrays or wide numeric identifiers;
#' preserve exact JSON where those distinctions matter.
#'
#' @param response Complete scenario-grid response as a nested list or one JSON string. Use
#'   existing JSON text when full-width integer identifiers must retain their exact tokens.
#' @return A newly allocated raw vector containing a native Arrow IPC file. Rows retain
#'   their scenario comparison order; nested results and errors use the native JSON
#'   columns.
#' @section Errors:
#' Malformed responses, unsupported exports, native allocation failures and invalid scalar-string inputs raise errors. Native errors raise R conditions; dataset errors with
#' validation details have class innovate_dataset_validation_error and carry code and
#' validation. JSON conversion, library loading and ABI errors propagate.
kernel_scenario_comparisons_ipc <- function(response) {
  scenario_output(response, innovate_scenario_comparisons_ipc_R)
}

#' Validate paired Bass trajectories and export their rows as Arrow IPC.
#'
#' Use exact JSON for full-width draw IDs. When reading with Arrow R, keep the table in
#' Arrow and cast draw_id to UTF-8 before conversion to ordinary R vectors if wide
#' identifiers must be preserved. This validates an interchange contract, not model
#' calibration.
#'
#' @param payload Paired-outcome contract with strategies and aligned draws, supplied as a
#'   nested list or a single JSON string. Draws include identifiers and time/adoption
#'   arrays; singleton trajectory arrays in lists are preserved explicitly.
#' @return A raw vector containing an Arrow IPC file with validated paired-outcome rows and
#'   native schema metadata.
#' @section Errors:
#' Native pairing, trajectory, value, identifier and output-allocation failures raise
#' errors. Native errors raise R conditions; dataset errors with validation details have
#' class innovate_dataset_validation_error and carry code and validation. JSON conversion,
#' library loading and ABI errors propagate.
kernel_paired_outcomes_ipc <- function(payload) {
  # Preserve trajectory arrays without changing scalar draw IDs or parameter values.
  if (is.list(payload) && is.list(payload[["strategies"]])) {
    for (i in seq_along(payload[["strategies"]])) {
      strategy <- payload[["strategies"]][[i]]
      if (!is.list(strategy) || !is.list(strategy[["draws"]])) next
      for (j in seq_along(strategy[["draws"]])) {
        draw <- strategy[["draws"]][[j]]
        if (!is.list(draw)) next
        for (field in c("time", "adoption")) {
          value <- draw[[field]]
          if (is.atomic(value) && length(value) == 1L && is.null(dim(value))) {
            draw[[field]] <- I(value)
          }
        }
        strategy[["draws"]][[j]] <- draw
      }
      payload[["strategies"]][[i]] <- strategy
    }
  }
  scenario_output(payload, innovate_paired_outcomes_ipc_R)
}

# Serialize an export list, or preserve supplied JSON/raw bytes, then call C.
# Successful Arrow output is already copied into an R raw vector by native.c;
# text output is decoded as a native error envelope.
scenario_output <- function(response, symbol) {
  if (is.list(response)) {
    response <- jsonlite::toJSON(response, auto_unbox = TRUE, null = "null", digits = I(17))
  }
  output <- .Call(symbol, native_library(), response)
  if (is.raw(output)) output else decode_kernel_response(output)
}

# Decode one native envelope without simplifying JSON arrays into R vectors.
# Structured dataset failures become innovate_dataset_validation_error conditions.
# Even raw=TRUE requests are inspected for errors before original text is returned.
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

#' Read the native model catalog and its capability metadata.
#'
#' Use the advertised operation and variant capabilities before constructing a request.
#' Presence in the catalog does not promise every simulation engine or constructor
#' combination.
#'
#' @return The discovery result as a named list containing models; the outer response
#'   envelope is omitted.
#' @section Errors:
#' Native errors raise R conditions; dataset errors with validation details have class
#' innovate_dataset_validation_error and carry code and validation. JSON conversion,
#' library loading and ABI errors propagate.
kernel_discover_models <- function() kernel_call("discover_models")$result
#' Fit a native model using observations and explicit fitting controls.
#'
#' A successful request does not establish global optimality, parameter identifiability or
#' forecast calibration. Inspect the native optimizer and diagnostic fields before
#' interpreting the fit.
#'
#' @param model_key Native model key returned by kernel_discover_models, such as
#'   "logistic".
#' @param payload Named list with inputs containing aligned time/t and observed/y arrays.
#'   Optional constructor_kwargs select a model variant and fit_options provide
#'   initial_parameters, weights and other supported controls. Multiseries layout is model-specific.
#' @return The full native envelope as a list. Its result includes fitted parameters,
#'   reusable state and model-specific optimizer/diagnostic fields. Retain the complete
#'   state for forecasting.
#' @section Errors:
#' Invalid observation shapes, unsupported variants, numerical constraints and native
#' fitting failures raise errors. Native errors raise R conditions; dataset errors with
#' validation details have class innovate_dataset_validation_error and carry code and
#' validation. JSON conversion, library loading and ABI errors propagate.
kernel_fit_model <- function(model_key, payload) kernel_call("fit_model", model_key, payload)
#' Predict native levels and rates from explicit parameters or fitted state.
#'
#' Levels and rates have different meanings. Deterministic predictions do not provide
#' calibrated intervals unless an appropriate separate inference operation supplies them.
#'
#' @param model_key Native model identifier matching the supplied parameters or state.
#' @param payload Named list with inputs.time or inputs.t and either parameters plus
#'   optional constructor_kwargs, or retained fitted state. Include required model-specific
#'   covariates and prediction options.
#' @return The full native envelope. The model-specific result may contain scalar values,
#'   multiseries rows, rates and state; JSON arrays remain nested lists in observation
#'   order.
#' @section Errors:
#' Unknown models, invalid grids, incompatible state/options, missing covariates and
#' numerical-domain violations raise errors. Native errors raise R conditions; dataset
#' errors with validation details have class innovate_dataset_validation_error and carry
#' code and validation. JSON conversion, library loading and ABI errors propagate.
kernel_predict_model <- function(model_key, payload) kernel_call("predict_model", model_key, payload)
#' Run a native stochastic model with an explicit simulation engine.
#'
#' Retain engine metadata, seed and inputs when recording an experiment. Replay is
#' qualified for a runtime and configuration; this helper does not guarantee cross-version
#' identity or empirical adequacy.
#'
#' @param model_key Native model key that supports the selected simulation engine.
#' @param payload Named list containing inputs, parameters and simulation_options such as
#'   engine, seed, population, network structure and timed events; constructor_kwargs are
#'   model-specific.
#' @return The complete native envelope with engine-specific trajectories and metadata.
#'   Count-based engines may return both integer counts and scaled values.
#' @section Errors:
#' Unsupported engines or model/engine combinations, invalid seeds/populations/events and
#' numerical limits raise errors. Native errors raise R conditions; dataset errors with
#' validation details have class innovate_dataset_validation_error and carry code and
#' validation. JSON conversion, library loading and ABI errors propagate.
kernel_simulate_model <- function(model_key, payload) kernel_call("simulate_model", model_key, payload)
#' Request a model-specific native analytical or numerical summary.
#'
#' Summary quantities inherit the model assumptions and supplied units; availability is not
#' evidence of causal or forecast validity.
#'
#' @param model_key Native model identifier defining the supported summary contract.
#' @param payload Named list of parameters or retained state and any model-specific
#'   constructor, threshold or time-grid inputs required by the summary.
#' @return The complete response envelope with model-specific summary fields in result. The
#'   wrapper does not coerce distinct models into a common table.
#' @section Errors:
#' Unsupported summaries, missing inputs, incompatible state and numerical-domain
#' violations raise errors. Native errors raise R conditions; dataset errors with
#' validation details have class innovate_dataset_validation_error and carry code and
#' validation. JSON conversion, library loading and ABI errors propagate.
kernel_summarize_model <- function(model_key, payload) kernel_call("summarize_model", model_key, payload)
#' Compute native diagnostics for observations and a supplied model state.
#'
#' Interpret metrics with their sample and model assumptions. In-sample diagnostics alone
#' do not establish out-of-sample calibration or causality.
#'
#' @param model_key Native model identifier for the observed data and prediction state.
#' @param payload Model-specific named list containing observation inputs and the
#'   parameters or state needed for prediction and diagnostic comparison.
#' @return The complete envelope with native diagnostic metrics and details in result;
#'   metric availability depends on the model.
#' @section Errors:
#' Misaligned observations, unsupported diagnostics, incompatible state and numerical-domain problems raise errors. Native errors raise R conditions; dataset errors with
#' validation details have class innovate_dataset_validation_error and carry code and
#' validation. JSON conversion, library loading and ABI errors propagate.
kernel_diagnose_model <- function(model_key, payload) kernel_call("diagnose_model", model_key, payload)

#' Fit independent payloads and retain per-item failures.
#'
#' Worker count controls concurrency, not the ordering or scientific interpretation of
#' results.
#'
#' @param model_key Native model identifier shared by every dataset.
#' @param datasets Ordered list of fit_model payloads, each containing its own inputs and
#'   fitting options or parameters/state. These are payloads, not full request envelopes.
#' @param workers Integer worker count from 1 through 32, default 1L. Parallel execution
#'   preserves input order.
#' @return An ordered list of result/error records, one per dataset. Each item retains an
#'   individual native failure instead of raising it at the batch boundary.
#' @section Errors:
#' An invalid outer batch, model-key shape or worker count raises a native error.
#' Serialization and library errors propagate; callers must inspect each returned item
#' error.
kernel_fit_batch <- function(model_key, datasets, workers = 1L) kernel_call("fit_batch", model_key, list(datasets = datasets, workers = workers))$result$datasets
#' Predict independent payloads and retain per-item failures.
#'
#' Worker count controls concurrency, not the ordering or scientific interpretation of
#' results.
#'
#' @param model_key Native model identifier shared by every dataset.
#' @param datasets Ordered list of predict_model payloads, each containing its own inputs
#'   and fitting options or parameters/state. These are payloads, not full request
#'   envelopes.
#' @param workers Integer worker count from 1 through 32, default 1L. Parallel execution
#'   preserves input order.
#' @return An ordered list of result/error records, one per dataset. Each item retains an
#'   individual native failure instead of raising it at the batch boundary.
#' @section Errors:
#' An invalid outer batch, model-key shape or worker count raises a native error.
#' Serialization and library errors propagate; callers must inspect each returned item
#' error.
kernel_predict_batch <- function(model_key, datasets, workers = 1L) kernel_call("predict_batch", model_key, list(datasets = datasets, workers = workers))$result$datasets

#' Assign historical categories from sample-weighted observation times.
#'
#' Weighted time mean and standard deviation determine the five historical labels. Rates
#' are not multiplied by time-step widths. This categorizes sampled times rather than
#' individual people.
#'
#' @param time Nonempty finite numeric vector in strictly increasing order and one
#'   consistent unit.
#' @param rates Aligned finite nonnegative weights with positive total mass. Signed net
#'   adoption rates cannot be used directly as these weights.
#' @return A named list of time, adoption_rate and category arrays. Every time receives one
#'   label, including zero-weight samples; boundary ties use the earlier category.
#' @section Errors:
#' Empty or misaligned arrays, nonfinite values, non-increasing times, negative rates and
#' zero total weight raise native errors. Serialization/library errors also propagate.
kernel_categorize_adopters <- function(time, rates) {
  kernel_call("categorize_adopters", payload = list(time = time, adoption_rate = rates))$result
}

#' Predict a single native rate series and apply historical categories.
#'
#' The helper flattens only one rate series. Multiseries outputs and negative substitution
#' flows require an explicit scientific choice of admissible weights outside this function.
#'
#' @param model_key Native model key whose prediction supplies one rate per observation
#'   time.
#' @param payload Prediction payload with inputs.time or inputs.t and parameters or
#'   retained state, as accepted by kernel_predict_model.
#' @return The time, adoption_rate and category result list returned by
#'   kernel_categorize_adopters.
#' @section Errors:
#' An absent rates field or rows with other than one rate raise an R error. Prediction
#' errors and the nonnegative/strict-time categorization validation propagate.
kernel_categorize_fitted_adopters <- function(model_key, payload) {
  rates <- kernel_predict_model(model_key, payload)$result$rates
  if (is.null(rates) || !all(lengths(rates) == 1L)) {
    stop("adopter classification requires one native rate per time", call. = FALSE)
  }
  time <- payload$inputs$time
  if (is.null(time)) time <- payload$inputs$t
  kernel_categorize_adopters(time, unlist(rates, use.names = FALSE))
}

#' Compute native cumulative totals in the supplied observation order.
#'
#' This calls native preprocessing rather than R cumsum, and returns the decoder's list
#' representation of JSON arrays.
#'
#' @param values Finite numeric vector without missing entries; no sorting, timestamp
#'   inference or imputation is performed.
#' @return A list of numeric cumulative totals with the same length as values; an empty
#'   input yields an empty list.
#' @section Errors:
#' Nonnumeric/nonfinite samples or cumulative totals outside the finite numerical domain
#' raise native errors. JSON/library failures propagate.
kernel_cumulative_sum <- function(values) {
  kernel_call("preprocess_series", payload = list(method = "cumulative_sum", values = values))$result$values
}

#' Compute native trailing means after a complete sample window exists.
#'
#' The kernel uses guarded accumulation for finite values. The helper neither imputes
#' missing samples nor constructs calendar windows.
#'
#' @param values Finite numeric vector in observation order without missing values.
#' @param window Positive integer sample count per trailing window. A window longer than
#'   the series produces only unfilled positions.
#' @return A list aligned with values: leading entries are NULL until window samples exist,
#'   followed by numeric complete-window means.
#' @section Errors:
#' Nonpositive/noninteger windows, invalid samples or nonfinite arithmetic raise native
#' errors. JSON/library failures propagate.
kernel_rolling_average <- function(values, window) {
  kernel_call("preprocess_series", payload = list(method = "rolling_average", values = values, window = window))$result$values
}

#' Sum observations into calendar or earliest-sample weekly buckets.
#'
#' Offset timestamps normalize to UTC. Weekly timestamp buckets span seven days from the
#' earliest timestamp; other frequencies use the normalized date. Use kernel_call with
#' aggregate_dates for optional fiscal_start_month controls.
#'
#' @param dates Date, POSIXt or character vector aligned to values. POSIXt inputs format in
#'   UTC with six fractional-second digits; other inputs use as.character and must contain
#'   consistent ISO dates or timestamps.
#' @param values Aligned finite numeric values to sum; observations need not already be
#'   sorted.
#' @param frequency One of "D", "W", "M", "ME", "MS", "QS", "QE", "YS" or "YE".
#'   Quarter/year boundaries start in January in this wrapper.
#' @return A list containing sorted dates labels and summed values. Weekly buckets start at
#'   the earliest observation and omit empty weeks; other frequencies fill intervening
#'   periods with zero. End variants label period ends.
#' @section Errors:
#' Misaligned arrays, malformed or mixed timestamp kinds, unsupported frequencies,
#' nonfinite sums, date limits or more than 100000 output periods raise native errors.
#' Conversion/library failures propagate.
kernel_aggregate_dates <- function(dates, values, frequency) {
  if (inherits(dates, "POSIXt")) {
    dates <- format(dates, "%Y-%m-%dT%H:%M:%OS6Z", tz = "UTC")
  }
  kernel_call("aggregate_dates", payload = list(dates = as.character(dates),
    values = values, frequency = frequency))$result
}

#' Decompose an evenly spaced series into native trend, seasonal and residual components.
#'
#' A decomposition is not a future forecast or a calibrated uncertainty estimate.
#'
#' @param values Finite numeric vector without missing values, in evenly spaced sample
#'   order. The caller must establish spacing because no timestamps are supplied.
#' @param period Integer samples per cycle, at least 2, with at least three complete cycles
#'   available.
#' @param trend_span Trend smoothing span, NULL by default. A numeric span is required
#'   unless historical=TRUE, which derives an odd span from period and seasonal span;
#'   native span bounds apply.
#' @param robust Logical residual-reweighting control; defaults to FALSE.
#' @param historical Logical switch for historical STL smoothing/reweighting rules;
#'   defaults to FALSE.
#' @param inner_iterations Integer inner smoothing passes from 1 through 100, default 5L.
#' @param outer_iterations Integer reweighting passes from 0 through 100; default 5L when
#'   isTRUE(robust), otherwise 0L. Positive counts require robust=TRUE.
#' @param seasonal_span Optional odd seasonal smoothing span of at least 3. NULL selects
#'   the native default: 7 in historical mode or at most 7 samples per phase otherwise.
#' @param low_pass_span Optional odd low-pass span of at least 3. NULL selects the smallest
#'   odd integer strictly above period.
#' @param seasonal_degree Seasonal local polynomial degree, 0L or 1L; defaults to 1L.
#' @param seasonal_jump Positive integer spacing of seasonal fitted points; defaults to 1L.
#' @param trend_degree Trend local polynomial degree, 0L or 1L; defaults to 1L.
#' @param trend_jump Positive integer spacing of trend fitted points; defaults to 1L.
#' @param low_pass_degree Low-pass local polynomial degree, 0L or 1L; defaults to 1L.
#' @param low_pass_jump Positive integer spacing of low-pass fitted points; defaults to 1L.
#' @return A named list with trend, seasonal and residual arrays aligned with the input.
#'   The arrays retain JSON-list representation; no data frame is constructed.
#' @section Errors:
#' Too few cycles, invalid span/iteration/degree/jump controls, invalid samples and
#' nonfinite estimates raise native errors. JSON/library failures propagate.
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

#' Fit a no-intercept native SARIMA model and return in-sample fitted values.
#'
#' Outside historical mode, leading missing fitted values reflect differencing warmup. A
#' returned fit alone does not establish an adequate time-series model.
#'
#' @param values Finite numeric vector without missing values on an ordered, evenly spaced
#'   grid. Timestamps are not checked by this wrapper.
#' @param order Length-three nonnegative integer vector c(p, d, q), within native order and
#'   sample-size limits.
#' @param seasonal_order Length-four nonnegative integer vector c(P, D, Q, period), default
#'   c(0L, 0L, 0L, 0L). Seasonal differencing requires a period of at least 2.
#' @param historical Logical control for historical diffuse-initialization predictions
#'   where supported; defaults to FALSE.
#' @return A numeric vector aligned with values. Native null warmup positions become
#'   NA_real_; coefficients, future forecasts and prediction intervals are not returned.
#' @section Errors:
#' Invalid orders, insufficient samples, stationarity or estimation failures, invalid
#' samples and conversion/library problems raise errors.
kernel_sarima_fit <- function(values, order, seasonal_order = c(0L, 0L, 0L, 0L), historical = FALSE) {
  fitted <- kernel_call("preprocess_series", payload = list(method = "sarima_fit",
    values = values, order = order, seasonal_order = seasonal_order, historical = historical))$result$fitted
  vapply(fitted, function(value) if (is.null(value)) NA_real_ else value, numeric(1))
}

#' Rewrite supported Arrow IPC input in portable file format.
#'
#' The output includes the file footer required for random-access interchange. Rewriting
#' does not promise byte identity with the input or change the scientific meaning of its
#' columns.
#'
#' @param data Raw vector containing an Arrow IPC file or stream. Both input encodings are
#'   accepted by the native reader.
#' @return A new raw vector containing an Arrow IPC file. The native buffer is copied and
#'   freed before returning.
#' @section Errors:
#' Non-raw data is rejected in R. Invalid or unsupported IPC, native allocation/conversion
#' failures and library/ABI errors propagate.
export_ipc_file <- function(data) {
  if (!is.raw(data)) stop("data must be raw IPC bytes", call. = FALSE)
  scenario_output(data, innovate_export_ipc_file_R)
}

#' Rewrite supported Arrow IPC input in portable stream format.
#'
#' Stream schema metadata is carried in the leading message. Rewriting does not promise
#' byte identity with the input or change the scientific meaning of its columns.
#'
#' @param data Raw vector containing an Arrow IPC file or stream. Both input encodings are
#'   accepted by the native reader.
#' @return A new raw vector containing an Arrow IPC stream. The native buffer is copied and
#'   freed before returning.
#' @section Errors:
#' Non-raw data is rejected in R. Invalid or unsupported IPC, native allocation/conversion
#' failures and library/ABI errors propagate.
export_ipc_stream <- function(data) {
  if (!is.raw(data)) stop("data must be raw IPC bytes", call. = FALSE)
  scenario_output(data, innovate_export_ipc_stream_R)
}

#' Resolve an extension, invoke a trusted R callback and validate its output.
#'
#' The callback runs in the current R process with its normal permissions. Manifest and
#' output validation do not sandbox it or prove scientific validity.
#'
#' @param callback Trusted function called with the resolved manifest and payload. Its
#'   return value must satisfy the selected extension-point contract.
#' @param selection Named list passed to resolve_extension_manifest, including name,
#'   expected_version and extension_point.
#' @param payload Application value passed unchanged to callback; this wrapper does not
#'   serialize it before invocation.
#' @return The validated native result without its outer envelope: a dataset artifact,
#'   resolved model aliases, a diagnostics summary or imported artifact result, according
#'   to the extension point.
#' @section Errors:
#' Native manifest/output validation errors and JSON/library failures propagate. Any
#' callback error is raised unchanged.
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
