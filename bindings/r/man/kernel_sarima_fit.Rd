\name{kernel_sarima_fit}
\alias{kernel_sarima_fit}
\title{Fit a native no-intercept SARIMA model}
\description{
Fit a no-intercept native SARIMA model and return in-sample fitted values.
}
\usage{
kernel_sarima_fit(values, order, seasonal_order = c(0L, 0L, 0L, 0L), historical = FALSE)
}
\arguments{
\item{values}{Finite numeric vector without missing values on an ordered, evenly spaced grid. Timestamps are not checked by this wrapper.}
\item{order}{Length-three nonnegative integer vector c(p, d, q), within native order and sample-size limits.}
\item{seasonal_order}{Length-four nonnegative integer vector c(P, D, Q, period), default c(0L, 0L, 0L, 0L). Seasonal differencing requires a period of at least 2.}
\item{historical}{Logical control for historical diffuse-initialization predictions where supported; defaults to FALSE.}
}
\value{
A numeric vector aligned with values. Native null warmup positions become NA_real_; coefficients, future forecasts and prediction intervals are not returned.
}
\details{
Fit a no-intercept native SARIMA model and return in-sample fitted values.

Outside historical mode, leading missing fitted values reflect differencing warmup. A
returned fit alone does not establish an adequate time-series model.

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
Invalid orders, insufficient samples, stationarity or estimation failures, invalid samples and conversion/library problems raise errors.
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}, \code{\link{kernel_scenario_grid_ipc}}}
