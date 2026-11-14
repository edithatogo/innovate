\name{kernel_paired_outcomes_ipc}
\alias{kernel_paired_outcomes_ipc}
\title{Export validated paired outcome trajectories}
\description{
Validate paired Bass trajectories and export their rows as Arrow IPC.
}
\usage{
kernel_paired_outcomes_ipc(payload)
}
\arguments{
\item{payload}{Paired-outcome contract with strategies and aligned draws, supplied as a nested list or a single JSON string. Draws include identifiers and time/adoption arrays; singleton trajectory arrays in lists are preserved explicitly.}
}
\value{
A raw vector containing an Arrow IPC file with validated paired-outcome rows and native schema metadata.
}
\details{
Validate paired Bass trajectories and export their rows as Arrow IPC.

Use exact JSON for full-width draw IDs. When reading with Arrow R, keep the table in
Arrow and cast draw_id to UTF-8 before conversion to ordinary R vectors if wide
identifiers must be preserved. This validates an interchange contract, not model
calibration.

Install the matching native shared library with the package, or set
\env{INNOVATE_NATIVE_LIBRARY} before running an example. These helpers do not
download or compile a runtime. Native JSON responses and Arrow byte outputs are
copied into R-owned memory before their native allocations are released.
}
\section{Errors}{
Native pairing, trajectory, value, identifier and output-allocation failures raise errors. Native errors raise R conditions; dataset errors with validation details have class innovate_dataset_validation_error and carry code and validation. JSON conversion, library loading and ABI errors propagate.
}
\seealso{\code{\link{kernel_call}}, \code{\link{kernel_predict_model}}, \code{\link{kernel_fit_model}}, \code{\link{kernel_scenario_grid_ipc}}}
