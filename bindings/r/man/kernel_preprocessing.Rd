\name{kernel_cumulative_sum}
\alias{kernel_cumulative_sum}
\alias{kernel_rolling_average}
\title{Compute native cumulative and rolling statistics}
\description{
Thin R wrappers over the Rust-owned schema-2.0 native contract. Each function
retains its documented return representation and error behavior; Python is not required.
}
\usage{
kernel_cumulative_sum(values)
kernel_rolling_average(values, window)
}
\arguments{
\item{values}{
\describe{
\item{\code{kernel_cumulative_sum}}{Finite numeric vector without missing entries; no sorting, timestamp inference or imputation is performed.}
\item{\code{kernel_rolling_average}}{Finite numeric vector in observation order without missing values.}
}
}
\item{window}{Positive integer sample count per trailing window. A window longer than the series produces only unfilled positions.}
}
\value{
\describe{
\item{\code{kernel_cumulative_sum}}{A list of numeric cumulative totals with the same length as values; an empty input yields an empty list.}
\item{\code{kernel_rolling_average}}{A list aligned with values: leading entries are NULL until window samples exist, followed by numeric complete-window means.}
}
}
\details{
\subsection{\code{kernel_cumulative_sum}}{
Compute native cumulative totals in the supplied observation order.

This calls native preprocessing rather than R cumsum, and returns the decoder's list
representation of JSON arrays.
}
\subsection{\code{kernel_rolling_average}}{
Compute native trailing means after a complete sample window exists.

The kernel uses guarded accumulation for finite values. The helper neither imputes
missing samples nor constructs calendar windows.
}

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
\describe{
\item{\code{kernel_cumulative_sum}}{Nonnumeric/nonfinite samples or cumulative totals outside the finite numerical domain raise native errors. JSON/library failures propagate.}
\item{\code{kernel_rolling_average}}{Nonpositive/noninteger windows, invalid samples or nonfinite arithmetic raise native errors. JSON/library failures propagate.}
}
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}, \code{\link{kernel_scenario_grid_ipc}}}
\examples{
stopifnot(identical(as.numeric(unlist(kernel_cumulative_sum(c(1, 2, 3)))), c(1, 3, 6)))
rolling <- kernel_rolling_average(c(1, 2, 3), 2L)
stopifnot(is.null(rolling[[1]]), rolling[[2]] == 1.5, rolling[[3]] == 2.5)
}
