\name{export_ipc_file}
\alias{export_ipc_file}
\title{Rewrite Arrow IPC into file format}
\description{
Rewrite supported Arrow IPC input in portable file format.
}
\usage{
export_ipc_file(data)
}
\arguments{
\item{data}{Raw vector containing an Arrow IPC file or stream. Both input encodings are accepted by the native reader.}
}
\value{
A new raw vector containing an Arrow IPC file. The native buffer is copied and freed before returning.
}
\details{
Rewrite supported Arrow IPC input in portable file format.

The output includes the file footer required for random-access interchange. Rewriting
does not promise byte identity with the input or change the scientific meaning of its
columns.

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
Non-raw data is rejected in R. Invalid or unsupported IPC, native allocation/conversion failures and library/ABI errors propagate.
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}, \code{\link{kernel_scenario_grid_ipc}}}
