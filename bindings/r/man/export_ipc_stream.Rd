\name{export_ipc_stream}
\alias{export_ipc_stream}
\title{Rewrite Arrow IPC bytes}
\usage{export_ipc_stream(data)}
\arguments{\item{data}{A raw vector containing an Arrow IPC file or stream.}}
\value{A raw vector containing a portable Arrow IPC stream.}
\description{Rewrite IPC bytes with the native Polars kernel for interchange, carrying custom schema metadata in the leading message.}
