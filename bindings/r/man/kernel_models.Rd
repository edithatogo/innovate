\name{kernel_fit_model}
\alias{kernel_fit_model}
\alias{kernel_predict_model}
\alias{kernel_simulate_model}
\alias{kernel_summarize_model}
\alias{kernel_diagnose_model}
\title{Fit, predict, simulate and diagnose native models}
\description{
Thin R wrappers over the Rust-owned schema-2.0 native contract. Each function
retains its documented return representation and error behavior; Python is not required.
}
\usage{
kernel_fit_model(model_key, payload)
kernel_predict_model(model_key, payload)
kernel_simulate_model(model_key, payload)
kernel_summarize_model(model_key, payload)
kernel_diagnose_model(model_key, payload)
}
\arguments{
\item{model_key}{
\describe{
\item{\code{kernel_fit_model}}{Native model key returned by kernel_discover_models, such as "logistic".}
\item{\code{kernel_predict_model}}{Native model identifier matching the supplied parameters or state.}
\item{\code{kernel_simulate_model}}{Native model key that supports the selected simulation engine.}
\item{\code{kernel_summarize_model}}{Native model identifier defining the supported summary contract.}
\item{\code{kernel_diagnose_model}}{Native model identifier for the observed data and prediction state.}
}
}
\item{payload}{
\describe{
\item{\code{kernel_fit_model}}{Named list with inputs containing aligned time/t and observed/y arrays. Optional constructor_kwargs select a model variant and fit_options provide initial_parameters, weights and other supported controls. Multiseries layout is model-specific.}
\item{\code{kernel_predict_model}}{Named list with inputs.time or inputs.t and either parameters plus optional constructor_kwargs, or retained fitted state. Include required model-specific covariates and prediction options.}
\item{\code{kernel_simulate_model}}{Named list containing inputs, parameters and simulation_options such as engine, seed, population, network structure and timed events; constructor_kwargs are model-specific.}
\item{\code{kernel_summarize_model}}{Named list of parameters or retained state and any model-specific constructor, threshold or time-grid inputs required by the summary.}
\item{\code{kernel_diagnose_model}}{Model-specific named list containing observation inputs and the parameters or state needed for prediction and diagnostic comparison.}
}
}
}
\value{
\describe{
\item{\code{kernel_fit_model}}{The full native envelope as a list. Its result includes fitted parameters, reusable state and model-specific optimizer/diagnostic fields. Retain the complete state for forecasting.}
\item{\code{kernel_predict_model}}{The full native envelope. The model-specific result may contain scalar values, multiseries rows, rates and state; JSON arrays remain nested lists in observation order.}
\item{\code{kernel_simulate_model}}{The complete native envelope with engine-specific trajectories and metadata. Count-based engines may return both integer counts and scaled values.}
\item{\code{kernel_summarize_model}}{The complete response envelope with model-specific summary fields in result. The wrapper does not coerce distinct models into a common table.}
\item{\code{kernel_diagnose_model}}{The complete envelope with native diagnostic metrics and details in result; metric availability depends on the model.}
}
}
\details{
\subsection{\code{kernel_fit_model}}{
Fit a native model using observations and explicit fitting controls.

A successful request does not establish global optimality, parameter identifiability or
forecast calibration. Inspect the native optimizer and diagnostic fields before
interpreting the fit.
}
\subsection{\code{kernel_predict_model}}{
Predict native levels and rates from explicit parameters or fitted state.

Levels and rates have different meanings. Deterministic predictions do not provide
calibrated intervals unless an appropriate separate inference operation supplies them.
}
\subsection{\code{kernel_simulate_model}}{
Run a native stochastic model with an explicit simulation engine.

Retain engine metadata, seed and inputs when recording an experiment. Replay is
qualified for a runtime and configuration; this helper does not guarantee cross-version
identity or empirical adequacy.
}
\subsection{\code{kernel_summarize_model}}{
Request a model-specific native analytical or numerical summary.

Summary quantities inherit the model assumptions and supplied units; availability is not
evidence of causal or forecast validity.
}
\subsection{\code{kernel_diagnose_model}}{
Compute native diagnostics for observations and a supplied model state.

Interpret metrics with their sample and model assumptions. In-sample diagnostics alone
do not establish out-of-sample calibration or causality.
}

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
\describe{
\item{\code{kernel_fit_model}}{Invalid observation shapes, unsupported variants, numerical constraints and native fitting failures raise errors. Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.}
\item{\code{kernel_predict_model}}{Unknown models, invalid grids, incompatible state/options, missing covariates and numerical-domain violations raise errors. Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.}
\item{\code{kernel_simulate_model}}{Unsupported engines or model/engine combinations, invalid seeds/populations/events and numerical limits raise errors. Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.}
\item{\code{kernel_summarize_model}}{Unsupported summaries, missing inputs, incompatible state and numerical-domain violations raise errors. Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.}
\item{\code{kernel_diagnose_model}}{Misaligned observations, unsupported diagnostics, incompatible state and numerical-domain problems raise errors. Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.}
}
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_scenario_grid_ipc}}}
\examples{
parameters <- list(L = 100, k = 0.5, x0 = 6)
time <- 0:12
prediction <- kernel_predict_model("logistic", list(
  inputs = list(time = time), parameters = parameters))
observed <- unlist(prediction$result$values, use.names = FALSE)
stopifnot(abs(observed[[7]] - 50) < 1e-12)
fitted <- kernel_fit_model("logistic", list(
  inputs = list(time = time, observed = observed),
  fit_options = list(initial_parameters = parameters)))
future <- kernel_predict_model("logistic", list(
  inputs = list(time = c(13, 14)), state = fitted$result$state))
stopifnot(length(future$result$values) == 2L)
}
