\name{kernel_call}
\alias{kernel_call}
\alias{kernel_discover_models}
\alias{kernel_fit_model}
\alias{kernel_predict_model}
\alias{kernel_simulate_model}
\alias{kernel_summarize_model}
\alias{kernel_diagnose_model}
\title{Invoke the Innovate Rust kernel}
\description{Native schema 2.0 requests with structured errors. No Python runtime is required.}
\usage{
kernel_call(operation, model_key = NULL, payload = list())
kernel_discover_models()
kernel_fit_model(model_key, payload)
kernel_predict_model(model_key, payload)
kernel_simulate_model(model_key, payload)
kernel_summarize_model(model_key, payload)
kernel_diagnose_model(model_key, payload)
}
\arguments{
\item{operation}{A canonical Rust kernel operation.}
\item{model_key}{A model key returned by native discovery.}
\item{payload}{Named list of inputs, parameters or fitted state.}
}
\value{A parsed native response. Unavailable features raise an error.}
\details{Set INNOVATE_NATIVE_LIBRARY to the installed native shared library. Advanced features and native distribution packaging remain under development.}
