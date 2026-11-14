\name{invoke_registered_extension}
\alias{invoke_registered_extension}
\title{Invoke a trusted registered extension}
\description{
Resolve an extension, invoke a trusted R callback and validate its output.
}
\usage{
invoke_registered_extension(callback, selection, payload)
}
\arguments{
\item{callback}{Trusted function called with the resolved manifest and payload. Its return value must satisfy the selected extension-point contract.}
\item{selection}{Named list passed to resolve_extension_manifest, including name, expected_version and extension_point.}
\item{payload}{Application value passed unchanged to callback; this wrapper does not serialize it before invocation.}
}
\value{
The validated native result without its outer envelope: a dataset artifact, resolved model aliases, a diagnostics summary or imported artifact result, according to the extension point.
}
\details{
Resolve an extension, invoke a trusted R callback and validate its output.

The callback runs in the current R process with its normal permissions. Manifest and
output validation do not sandbox it or prove scientific validity.

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
Native manifest/output validation errors and JSON/library failures propagate. Any callback error is raised unchanged.
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}, \code{\link{kernel_scenario_grid_ipc}}}
