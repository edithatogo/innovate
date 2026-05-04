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
\title{Innovate kernel bridge helpers}
\description{
Construct Innovate kernel requests, invoke the shared Python kernel bridge, and
convert kernel responses into ordinary R objects.
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
\item{operation}{A non-empty scalar character kernel operation name, such as
\code{"discover_models"}, \code{"fit_model"}, \code{"predict_model"},
\code{"simulate_model"}, \code{"summarize_model"}, or
\code{"diagnose_model"}.}
\item{model_key}{A scalar character model key. Required for all operations
except \code{"discover_models"}.}
\item{payload}{A named list containing the operation-specific request payload.}
\item{metadata}{A named list containing optional request metadata.}
\item{schema_version}{A scalar character kernel schema version. Defaults to the
package's active schema version.}
\item{request}{A kernel request list, normally created by
\code{kernel_request()}.}
\item{response}{A decoded kernel response list, normally returned by
\code{kernel_call()}.}
\item{result}{A converted kernel result object returned by
\code{kernel_response_to_r()} or one of the operation helpers.}
}
\value{
\itemize{
\item \code{kernel_schema_version()} returns a scalar character schema version.
\item \code{kernel_request()} returns a request list with
\code{schema_version}, \code{operation}, \code{model_key}, \code{payload}, and
\code{metadata} fields.
\item \code{kernel_call()} returns a decoded response list from the Python
bridge.
\item \code{kernel_response_to_r()} and the operation helpers return converted
R objects. Kernel arrays become arrays, table-shaped payloads become data
frames, discovery payloads become data frames, and nested payloads remain named
lists.
\item \code{kernel_extract_diagnostics()} returns a diagnostics list when the
result exposes one.
}
}
\details{
The R package is intentionally a thin adapter over the stable Innovate
functional kernel contract. It does not reimplement model semantics. Execution
is delegated to the shared Python kernel bridge, while this package handles
request construction, process dispatch, error conversion, and response
conversion.

\code{kernel_call()} writes a temporary JSON request, invokes the bridge script,
and reads the JSON response. By default it uses \command{uv run python} when
\command{uv} is available, otherwise it falls back to \command{python3}. Set the
\env{INNOVATE_PYTHON_COMMAND} environment variable to override the launcher.

If the kernel response contains an error payload, \code{kernel_response_to_r()}
signals an \code{innovate_kernel_error} condition. The condition object contains
the original kernel error payload in its \code{error} field and the full
response in its \code{response} field.
}
\examples{
kernel_schema_version()

request <- kernel_request(
  operation = "predict_model",
  model_key = "logistic",
  payload = list(
    state = list(
      model_key = "logistic",
      parameters = list(L = 1, k = 0.5, x0 = 2)
    ),
    inputs = list(time = c(0, 1, 2))
  )
)
request$operation
request$model_key

response <- list(
  schema_version = "1.0",
  operation = "predict_model",
  result = list(shape = list(3L), dtype = "float64", values = list(0.1, 0.2, 0.4)),
  error = NULL,
  metadata = list()
)
values <- kernel_response_to_r(response)
values
attr(values, "kernel_operation")

summary_result <- list(
  diagnostics = list(
    support_level = "supported",
    provenance = "deterministic"
  )
)
kernel_extract_diagnostics(summary_result)
}
\seealso{
\code{\link{innovate.R-package}}
}
