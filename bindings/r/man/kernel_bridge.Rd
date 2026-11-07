\name{kernel_call}
\alias{kernel_call}
\alias{kernel_call_json}
\alias{kernel_scenario_grid_ipc}
\alias{kernel_scenario_executions_ipc}
\alias{kernel_scenario_comparisons_ipc}
\alias{kernel_paired_outcomes_ipc}
\alias{kernel_discover_models}
\alias{kernel_fit_model}
\alias{kernel_predict_model}
\alias{kernel_simulate_model}
\alias{kernel_summarize_model}
\alias{kernel_diagnose_model}
\alias{kernel_fit_batch}
\alias{kernel_predict_batch}
\alias{kernel_categorize_adopters}
\alias{kernel_categorize_fitted_adopters}
\alias{kernel_cumulative_sum}
\alias{kernel_rolling_average}
\alias{kernel_aggregate_dates}
\alias{kernel_seasonal_decomposition}
\alias{kernel_sarima_fit}
\title{Invoke the Innovate Rust kernel}
\description{Native schema 2.0 requests with structured errors. No Python runtime is required.}
\usage{
kernel_call(operation, model_key = NULL, payload = list())
kernel_call_json(request, raw = FALSE)
kernel_scenario_grid_ipc(data, payload)
kernel_scenario_executions_ipc(response)
kernel_scenario_comparisons_ipc(response)
kernel_paired_outcomes_ipc(payload)
kernel_discover_models()
kernel_fit_model(model_key, payload)
kernel_predict_model(model_key, payload)
kernel_simulate_model(model_key, payload)
kernel_summarize_model(model_key, payload)
kernel_diagnose_model(model_key, payload)
}
\arguments{
\item{operation}{A canonical Rust kernel operation.}
\item{model_key}{A model key returned by native discovery.}
\item{payload}{Named list of inputs, parameters or fitted state. The paired outcome export also accepts JSON text.}
\item{data}{A raw vector containing an Arrow IPC file with a numeric time column.}
\item{response}{A grid response list or JSON text. Use JSON text to retain full-width integers.}
\item{request}{A single JSON request string, passed unchanged to Rust.}
\item{raw}{Return the successful response as JSON text to retain exact integer tokens and array shapes.}
}
\value{A parsed native response, or JSON text when requested. Scenario and paired outcome exports return raw Arrow IPC bytes. Native errors raise an R error.}
\details{The batch functions take a model key, an ordered list of datasets
and a worker count. Adopter categorization takes aligned times and rates
or a fitted-model payload. Preprocessing helpers provide cumulative sums,
rolling averages, date aggregation, seasonal decomposition and SARIMA
fitted values. Set INNOVATE_NATIVE_LIBRARY to the installed native shared library. Advanced features and native distribution packaging remain under development.}
