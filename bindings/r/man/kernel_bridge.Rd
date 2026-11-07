\name{kernel_call}
\alias{kernel_call}
\alias{kernel_call_json}
\alias{kernel_scenario_grid_ipc}
\alias{kernel_scenario_executions_ipc}
\alias{kernel_scenario_comparisons_ipc}
\alias{kernel_discover_models}
\alias{kernel_fit_model}
\alias{kernel_predict_model}
\alias{kernel_simulate_model}
\alias{kernel_summarize_model}
\alias{kernel_diagnose_model}
\title{Invoke the Innovate Rust kernel}
\description{Native schema 2.0 requests with structured errors. No Python runtime is required.}
\usage{
kernel_call(operation, model_key = NULL, payload = list())
kernel_call_json(request, raw = FALSE)
kernel_scenario_grid_ipc(data, payload)
kernel_scenario_executions_ipc(response)
kernel_scenario_comparisons_ipc(response)
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
\item{payload}{Named list of inputs, parameters or fitted state.}
\item{data}{A raw vector containing an Arrow IPC file with a numeric time column.}
\item{response}{A grid response list or JSON text. Use JSON text to retain full-width integers.}
\item{request}{A single JSON request string, passed unchanged to Rust.}
\item{raw}{Return the successful response as JSON text to retain exact integer tokens and array shapes.}
}
\value{A parsed native response, or JSON text when requested. Scenario execution and comparison exports return raw Arrow IPC bytes. Native errors raise an R error.}
\details{Set INNOVATE_NATIVE_LIBRARY to the installed native shared library. Advanced features and native distribution packaging remain under development.}
