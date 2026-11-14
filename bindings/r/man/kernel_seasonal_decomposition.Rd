\name{kernel_seasonal_decomposition}
\alias{kernel_seasonal_decomposition}
\title{Decompose an evenly spaced seasonal series}
\description{
Decompose an evenly spaced series into native trend, seasonal and residual components.
}
\usage{
kernel_seasonal_decomposition(values, period, trend_span = NULL, robust = FALSE, historical = FALSE,
                                          inner_iterations = 5L, outer_iterations = if (isTRUE(robust)) 5L else 0L,
                                          seasonal_span = NULL, low_pass_span = NULL,
                                          seasonal_degree = 1L, seasonal_jump = 1L,
                                          trend_degree = 1L, trend_jump = 1L,
                                          low_pass_degree = 1L, low_pass_jump = 1L)
}
\arguments{
\item{values}{Finite numeric vector without missing values, in evenly spaced sample order. The caller must establish spacing because no timestamps are supplied.}
\item{period}{Integer samples per cycle, at least 2, with at least three complete cycles available.}
\item{trend_span}{Trend smoothing span, NULL by default. A numeric span is required unless historical=TRUE, which derives an odd span from period and seasonal span; native span bounds apply.}
\item{robust}{Logical residual-reweighting control; defaults to FALSE.}
\item{historical}{Logical switch for historical STL smoothing/reweighting rules; defaults to FALSE.}
\item{inner_iterations}{Integer inner smoothing passes from 1 through 100, default 5L.}
\item{outer_iterations}{Integer reweighting passes from 0 through 100; default 5L when isTRUE(robust), otherwise 0L. Positive counts require robust=TRUE.}
\item{seasonal_span}{Optional odd seasonal smoothing span of at least 3. NULL selects the native default: 7 in historical mode or at most 7 samples per phase otherwise.}
\item{low_pass_span}{Optional odd low-pass span of at least 3. NULL selects the smallest odd integer strictly above period.}
\item{seasonal_degree}{Seasonal local polynomial degree, 0L or 1L; defaults to 1L.}
\item{seasonal_jump}{Positive integer spacing of seasonal fitted points; defaults to 1L.}
\item{trend_degree}{Trend local polynomial degree, 0L or 1L; defaults to 1L.}
\item{trend_jump}{Positive integer spacing of trend fitted points; defaults to 1L.}
\item{low_pass_degree}{Low-pass local polynomial degree, 0L or 1L; defaults to 1L.}
\item{low_pass_jump}{Positive integer spacing of low-pass fitted points; defaults to 1L.}
}
\value{
A named list with trend, seasonal and residual arrays aligned with the input. The arrays retain JSON-list representation; no data frame is constructed.
}
\details{
Decompose an evenly spaced series into native trend, seasonal and residual components.

A decomposition is not a future forecast or a calibrated uncertainty estimate.

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
Too few cycles, invalid span/iteration/degree/jump controls, invalid samples and nonfinite estimates raise native errors. JSON/library failures propagate.
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}, \code{\link{kernel_scenario_grid_ipc}}}
