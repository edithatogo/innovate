\name{kernel_categorize_adopters}
\alias{kernel_categorize_adopters}
\alias{kernel_categorize_fitted_adopters}
\title{Classify sample-weighted adopter times}
\description{
Thin R wrappers over the Rust-owned schema-2.0 native contract. Each function
retains its documented return representation and error behavior; Python is not required.
}
\usage{
kernel_categorize_adopters(time, rates)
kernel_categorize_fitted_adopters(model_key, payload)
}
\arguments{
\item{time}{Nonempty finite numeric vector in strictly increasing order and one consistent unit.}
\item{rates}{Aligned finite nonnegative weights with positive total mass. Signed net adoption rates cannot be used directly as these weights.}
\item{model_key}{Native model key whose prediction supplies one rate per observation time.}
\item{payload}{Prediction payload with inputs.time or inputs.t and parameters or retained state, as accepted by kernel_predict_model.}
}
\value{
\describe{
\item{\code{kernel_categorize_adopters}}{A named list of time, adoption_rate and category arrays. Every time receives one label, including zero-weight samples; boundary ties use the earlier category.}
\item{\code{kernel_categorize_fitted_adopters}}{The time, adoption_rate and category result list returned by kernel_categorize_adopters.}
}
}
\details{
\subsection{\code{kernel_categorize_adopters}}{
Assign historical categories from sample-weighted observation times.

Weighted time mean and standard deviation determine the five historical labels. Rates
are not multiplied by time-step widths. This categorizes sampled times rather than
individual people.
}
\subsection{\code{kernel_categorize_fitted_adopters}}{
Predict a single native rate series and apply historical categories.

The helper flattens only one rate series. Multiseries outputs and negative substitution
flows require an explicit scientific choice of admissible weights outside this function.
}

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
\describe{
\item{\code{kernel_categorize_adopters}}{Empty or misaligned arrays, nonfinite values, non-increasing times, negative rates and zero total weight raise native errors. Serialization/library errors also propagate.}
\item{\code{kernel_categorize_fitted_adopters}}{An absent rates field or rows with other than one rate raise an R error. Prediction errors and the nonnegative/strict-time categorization validation propagate.}
}
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}, \code{\link{kernel_scenario_grid_ipc}}}
\examples{
result <- kernel_categorize_adopters(0:4, c(0, 3, 3, 1, 1))
stopifnot(identical(unlist(result$category, use.names = FALSE),
  c("Innovators", "Early Adopters", "Early Majority", "Late Majority", "Laggards")))
}
