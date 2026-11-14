\name{kernel_call}
\alias{kernel_call}
\alias{kernel_call_json}
\alias{kernel_discover_models}
\title{Invoke the native Innovate JSON kernel}
\description{
Thin R wrappers over the Rust-owned schema-2.0 native contract. Each function
retains its documented return representation and error behavior; Python is not required.
}
\usage{
kernel_call(operation, model_key = NULL, payload = list(), raw = FALSE)
kernel_call_json(request, raw = FALSE)
kernel_discover_models()
}
\arguments{
\item{operation}{Canonical native operation name, such as "discover_models" or "predict_model".}
\item{model_key}{Native model key; NULL by default for operations without a selected model.}
\item{payload}{Operation-specific named list, defaulting to an empty JSON object. Known native array fields retain array shape even for a single element; scalar parameter fields remain scalars. NULL values are encoded as JSON null.}
\item{raw}{
\describe{
\item{\code{kernel_call}}{One non-missing logical value, FALSE by default. TRUE returns successful response JSON without replacing its original numeric tokens or array shapes.}
\item{\code{kernel_call_json}}{Single non-missing logical value, FALSE by default. TRUE returns the original successful response string after native-error inspection.}
}
}
\item{request}{Single non-missing character string containing the full schema-2.0 envelope: schema_version, operation, optional model_key and payload.}
}
\value{
\describe{
\item{\code{kernel_call}}{A complete native response as nested lists when raw is FALSE, or the original response string when TRUE. Arrays decode as lists and JSON null as NULL. Successful envelopes may omit error; non-null native errors are raised in either mode.}
\item{\code{kernel_call_json}}{The parsed native envelope as nested lists, or its exact JSON string if raw is TRUE. Even raw responses are checked for a native error before being returned.}
\item{\code{kernel_discover_models}}{The discovery result as a named list containing models; the outer response envelope is omitted.}
}
}
\details{
\subsection{\code{kernel_call}}{
Execute a schema-2.0 request in the native Rust kernel.

Requests are encoded with jsonlite using 17 digits for numeric values. R numeric
precision already lost before encoding cannot be recovered; use kernel_call_json for
exact existing JSON. Scientific computation and validation run in Rust with no Python
fallback. JSON calls keep a loader reference for process-local registry state, copy
response text into R, and release native response memory.
}
\subsection{\code{kernel_call_json}}{
Send an existing JSON request without re-encoding its numbers or arrays.

Use this entry point for full-width integer identifiers or externally prepared schema
arrays. Parsed R numeric values may round wide integers; raw=TRUE preserves their
response tokens. INNOVATE_NATIVE_LIBRARY overrides the installed platform-specific
library in the package lib directory.
}
\subsection{\code{kernel_discover_models}}{
Read the native model catalog and its capability metadata.

Use the advertised operation and variant capabilities before constructing a request.
Presence in the catalog does not promise every simulation engine or constructor
combination.
}

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
\describe{
\item{\code{kernel_call}}{Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate. An invalid raw flag is rejected by kernel_call_json.}
\item{\code{kernel_call_json}}{Invalid request/raw types, lengths or missing values are rejected before the native call. Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.}
\item{\code{kernel_discover_models}}{Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.}
}
}
\seealso{\code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}, \code{\link{kernel_scenario_grid_ipc}}}
\examples{
catalog <- kernel_discover_models()
stopifnot(any(vapply(catalog$models, function(model) model$key == "bass", logical(1))))
request <- '{"schema_version":"2.0","operation":"discover_models","payload":{}}'
response_json <- kernel_call_json(request, raw = TRUE)
stopifnot(is.character(response_json), length(response_json) == 1L)
}
