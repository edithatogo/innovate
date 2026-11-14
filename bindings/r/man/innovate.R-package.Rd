\name{innovate.R-package}
\alias{innovate.R-package}
\alias{innovate.R}
\docType{package}
\title{R bindings for the native Innovate kernel}
\description{
Thin R access to the Rust-owned scientific kernel through schema-2.0 JSON
requests and native Arrow IPC. No Python interpreter is used. Model fitting,
prediction, simulation and validation follow the native model-specific contract.
}
\details{
Install the matching native shared library in the package's platform library
directory, or set \env{INNOVATE_NATIVE_LIBRARY}. Loading the package does not
build or download that library. \code{kernel_discover_models} reports current
model keys and capabilities; \code{kernel_call} exposes native operations that
do not have a dedicated R convenience function.

Parsed JSON arrays remain lists, and nulls remain \code{NULL}. Use
\code{kernel_call_json(..., raw = TRUE)} to preserve exact existing response
tokens for full-width integer identifiers. Only \code{kernel_sarima_fit}
converts its selected fitted array to a numeric vector with \code{NA_real_}
for native null warmup entries.

The manual describes the current source interface. Structural documentation
coverage does not establish numerical accuracy, platform distribution
qualification or successful example execution. These require the native R test
suite and the corresponding runtime/build evidence.
}
\seealso{
\code{\link{kernel_call}}, \code{\link{kernel_call_json}},
\code{\link{kernel_discover_models}}, \code{\link{kernel_fit_model}},
\code{\link{kernel_predict_model}}, \code{\link{kernel_simulate_model}},
\code{\link{kernel_scenario_grid_ipc}}
}
