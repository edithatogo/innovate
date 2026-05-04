\name{kernel_bridge}
\alias{kernel_bridge}
\alias{kernel_call}
\alias{kernel_diagnose_model}
\alias{kernel_discover_models}
\alias{kernel_extract_diagnostics}
\alias{kernel_fit_model}
\alias{kernel_predict_model}
\alias{kernel_request}
\alias{kernel_response_to_r}
\alias{kernel_schema_version}
\alias{kernel_simulate_model}
\alias{kernel_summarize_model}
\title{Thin R bindings for the Innovate kernel}
\description{
Helpers for constructing Innovate kernel requests, invoking the shared Python
kernel bridge, and converting kernel responses into R objects.
}
\usage{
kernel_schema_version()
kernel_request(operation, model_key = NULL, payload = list(), metadata = list(),
  schema_version = KERNEL_SCHEMA_VERSION)
kernel_call(request)
kernel_response_to_r(response)
kernel_discover_models()
kernel_fit_model(request)
kernel_predict_model(request)
kernel_simulate_model(request)
kernel_summarize_model(request)
kernel_diagnose_model(request)
kernel_extract_diagnostics(result)
}
\arguments{
\item{operation}{Kernel operation name.}
\item{model_key}{Optional model key required for non-discovery operations.}
\item{payload}{Request payload list.}
\item{metadata}{Request metadata list.}
\item{schema_version}{Kernel schema version.}
\item{request}{Kernel request list.}
\item{response}{Kernel response list.}
\item{result}{Converted kernel result.}
}
\value{
Converted kernel responses, request lists, diagnostics lists, or the active
kernel schema version string depending on the helper.
}
\details{
The R package is intentionally a thin adapter. It does not reimplement model
semantics; it delegates execution to the shared Innovate functional kernel
bridge.
}
\examples{
kernel_schema_version()

request <- kernel_request(
  operation = "fit_model",
  model_key = "bass",
  payload = list(inputs = list(time = c(0, 1), observed = c(0.1, 0.2)))
)
request$operation

response <- list(
  schema_version = "1.0",
  operation = "predict_model",
  result = list(shape = list(2L), dtype = "float64", values = list(0.1, 0.2)),
  error = NULL,
  metadata = list()
)
kernel_response_to_r(response)
}
