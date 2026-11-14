\name{kernel_fit_batch}
\alias{kernel_fit_batch}
\alias{kernel_predict_batch}
\title{Execute ordered native model batches}
\description{
Thin R wrappers over the Rust-owned schema-2.0 native contract. Each function
retains its documented return representation and error behavior; Python is not required.
}
\usage{
kernel_fit_batch(model_key, datasets, workers = 1L)
kernel_predict_batch(model_key, datasets, workers = 1L)
}
\arguments{
\item{model_key}{Native model identifier shared by every dataset.}
\item{datasets}{
\describe{
\item{\code{kernel_fit_batch}}{Ordered list of fit_model payloads, each containing its own inputs and fitting options or parameters/state. These are payloads, not full request envelopes.}
\item{\code{kernel_predict_batch}}{Ordered list of predict_model payloads, each containing its own inputs and fitting options or parameters/state. These are payloads, not full request envelopes.}
}
}
\item{workers}{Integer worker count from 1 through 32, default 1L. Parallel execution preserves input order.}
}
\value{
\describe{
\item{\code{kernel_fit_batch}}{An ordered list of result/error records, one per dataset. Each item retains an individual native failure instead of raising it at the batch boundary.}
\item{\code{kernel_predict_batch}}{An ordered list of result/error records, one per dataset. Each item retains an individual native failure instead of raising it at the batch boundary.}
}
}
\details{
\subsection{\code{kernel_fit_batch}}{
Fit independent payloads and retain per-item failures.

Worker count controls concurrency, not the ordering or scientific interpretation of
results.
}
\subsection{\code{kernel_predict_batch}}{
Predict independent payloads and retain per-item failures.

Worker count controls concurrency, not the ordering or scientific interpretation of
results.
}

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
\describe{
\item{\code{kernel_fit_batch}}{An invalid outer batch, model-key shape or worker count raises a native error. Serialization and library errors propagate; callers must inspect each returned item error.}
\item{\code{kernel_predict_batch}}{An invalid outer batch, model-key shape or worker count raises a native error. Serialization and library errors propagate; callers must inspect each returned item error.}
}
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}, \code{\link{kernel_scenario_grid_ipc}}}
\examples{
rows <- kernel_predict_batch("logistic", list(
  list(inputs = list(time = c(0, 1)), parameters = list(L = 100, k = 1, x0 = 0)),
  list(inputs = list(time = 0), parameters = list(L = 200, k = 1, x0 = 0))
), workers = 2L)
stopifnot(length(rows) == 2L, is.null(rows[[1]]$error), is.null(rows[[2]]$error))
stopifnot(rows[[1]]$result$values[[1]] == 50,
          rows[[2]]$result$values[[1]] == 100)
}
