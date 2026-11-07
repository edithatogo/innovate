\name{invoke_registered_extension}
\alias{invoke_registered_extension}
\title{Invoke a trusted extension callback}
\usage{invoke_registered_extension(callback, selection, payload)}
\arguments{
\item{callback}{A caller-supplied function of the resolved selection and payload.}
\item{selection}{A list with name, expected_version and extension_point.}
\item{payload}{The value passed unchanged to the callback.}
}
\description{Resolve a registered extension in Rust before invoking a trusted caller-supplied function.}
\value{The value returned by the callback.}
