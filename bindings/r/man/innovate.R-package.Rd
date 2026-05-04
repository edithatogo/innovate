\name{innovate.R-package}
\alias{innovate.R-package}
\alias{innovate.R}
\docType{package}
\title{R bindings for the Innovate kernel}
\description{
Thin R bindings for the Innovate functional kernel. The package exposes request
builders, bridge dispatch helpers, response conversion, and diagnostics helpers
without duplicating model logic in R.
}
\details{
The package is organized around the shared Innovate kernel schema. The exported
helpers build request lists, call the Python-backed kernel bridge, and convert
JSON responses into R-native arrays, data frames, lists, and conditions.

Use \code{\link{kernel_discover_models}()} to inspect available model metadata
when the bridge runtime is available. Use \code{\link{kernel_request}()} plus
one of the operation helpers for \code{fit_model}, \code{predict_model},
\code{simulate_model}, \code{summarize_model}, or \code{diagnose_model}
requests.

Examples in the reference manual avoid launching the bridge so they remain
deterministic in offline and CRAN-style checks.
}
\seealso{
\code{\link{kernel_schema_version}},
\code{\link{kernel_request}},
\code{\link{kernel_call}},
\code{\link{kernel_response_to_r}},
\code{\link{kernel_discover_models}},
\code{\link{kernel_fit_model}},
\code{\link{kernel_predict_model}},
\code{\link{kernel_simulate_model}},
\code{\link{kernel_summarize_model}},
\code{\link{kernel_diagnose_model}},
\code{\link{kernel_extract_diagnostics}}
}
\examples{
kernel_schema_version()

request <- kernel_request(
  operation = "summarize_model",
  model_key = "logistic",
  payload = list(inputs = list(time = c(0, 1), observed = c(0.2, 0.5)))
)
names(request)
}
