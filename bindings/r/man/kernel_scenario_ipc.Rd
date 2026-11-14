\name{kernel_scenario_grid_ipc}
\alias{kernel_scenario_grid_ipc}
\alias{kernel_scenario_executions_ipc}
\alias{kernel_scenario_comparisons_ipc}
\title{Execute and export native scenario grids}
\description{
Thin R wrappers over the Rust-owned schema-2.0 native contract. Each function
retains its documented return representation and error behavior; Python is not required.
}
\usage{
kernel_scenario_grid_ipc(data, payload)
kernel_scenario_executions_ipc(response)
kernel_scenario_comparisons_ipc(response)
}
\arguments{
\item{data}{Raw vector containing an Arrow IPC file or stream with the required numeric time column.}
\item{payload}{Named list containing ordered scenarios and optional comparisons/execution controls, not a complete schema envelope. Wrap singleton numeric arrays in I() when their fields require JSON arrays; this entry point directly uses jsonlite auto-unboxing.}
\item{response}{Complete scenario-grid response as a nested list or one JSON string. Use existing JSON text when full-width integer identifiers must retain their exact tokens.}
}
\value{
\describe{
\item{\code{kernel_scenario_grid_ipc}}{The decoded native response envelope as nested lists, including ordered execution rows and any requested comparisons/artifacts. Individual scenario failures remain in their rows; outer errors are raised.}
\item{\code{kernel_scenario_executions_ipc}}{A newly allocated raw vector containing a native Arrow IPC file. Rows retain their scenario execution order; nested results and errors use the native JSON columns.}
\item{\code{kernel_scenario_comparisons_ipc}}{A newly allocated raw vector containing a native Arrow IPC file. Rows retain their scenario comparison order; nested results and errors use the native JSON columns.}
}
}
\details{
\subsection{\code{kernel_scenario_grid_ipc}}{
Execute a scenario grid on a shared Arrow IPC time column.

The synchronous C call reads the supplied bytes; its JSON response is copied into R and
native allocation is released. The helper returns structured JSON, not an Arrow R table.
}
\subsection{\code{kernel_scenario_executions_ipc}}{
Encode ordered scenario execution rows as an Arrow IPC file.

The bridge copies output bytes into R and frees native memory before returning. Passing
a parsed list through jsonlite can alter singleton arrays or wide numeric identifiers;
preserve exact JSON where those distinctions matter.
}
\subsection{\code{kernel_scenario_comparisons_ipc}}{
Encode ordered scenario comparison rows as an Arrow IPC file.

The bridge copies output bytes into R and frees native memory before returning. Passing
a parsed list through jsonlite can alter singleton arrays or wide numeric identifiers;
preserve exact JSON where those distinctions matter.
}

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
\describe{
\item{\code{kernel_scenario_grid_ipc}}{The native bridge rejects non-raw input, invalid IPC or grid requests. Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.}
\item{\code{kernel_scenario_executions_ipc}}{Malformed responses, unsupported exports, native allocation failures and invalid scalar-string inputs raise errors. Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.}
\item{\code{kernel_scenario_comparisons_ipc}}{Malformed responses, unsupported exports, native allocation failures and invalid scalar-string inputs raise errors. Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.}
}
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}}
