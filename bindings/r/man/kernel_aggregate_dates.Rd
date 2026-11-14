\name{kernel_aggregate_dates}
\alias{kernel_aggregate_dates}
\title{Aggregate observations into calendar periods}
\description{
Sum observations into calendar or earliest-sample weekly buckets.
}
\usage{
kernel_aggregate_dates(dates, values, frequency)
}
\arguments{
\item{dates}{Date, POSIXt or character vector aligned to values. POSIXt inputs format in UTC with six fractional-second digits; other inputs use as.character and must contain consistent ISO dates or timestamps.}
\item{values}{Aligned finite numeric values to sum; observations need not already be sorted.}
\item{frequency}{One of "D", "W", "M", "ME", "MS", "QS", "QE", "YS" or "YE". Quarter/year boundaries start in January in this wrapper.}
}
\value{
A list containing sorted dates labels and summed values. Weekly buckets start at the earliest observation and omit empty weeks; other frequencies fill intervening periods with zero. End variants label period ends.
}
\details{
Sum observations into calendar or earliest-sample weekly buckets.

Offset timestamps normalize to UTC. Weekly timestamp buckets span seven days from the
earliest timestamp; other frequencies use the normalized date. Use kernel_call with
aggregate_dates for optional fiscal_start_month controls.

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
Misaligned arrays, malformed or mixed timestamp kinds, unsupported frequencies, nonfinite sums, date limits or more than 100000 output periods raise native errors. Conversion/library failures propagate.
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}, \code{\link{kernel_scenario_grid_ipc}}}
\examples{
weekly <- kernel_aggregate_dates(c("2026-11-01", "2026-11-02"), c(1, 2), "W")
stopifnot(weekly$dates[[1]] == "2026-11-01", weekly$values[[1]] == 3)
}
