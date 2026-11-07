# Run from the repository root with the paired historical export as the argument.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
fixture <- "specs/ecosystem/voiage/uncertainty/diffusion_v1"
expected <- read.csv(file.path(fixture, "adoption_trajectories.csv"))
actual <- arrow::read_ipc_file(args[[1L]])
stopifnot(
  identical(names(actual), c("scenario_id", "draw_id", "time_unit", "perspective",
    "provenance", "parameters_json", "time", "adoption")),
  identical(actual$scenario_id, expected$scenario_id),
  all(as.numeric(actual$draw_id) == expected$draw_id),
  all(actual$time == expected$time),
  max(abs(actual$adoption - expected$adoption)) < 1e-12,
  all(is.na(actual$perspective)),
  all(actual$time_unit == "period"),
  all(actual$provenance == "voiage_diffusion_uncertainty_v1")
)
cat(sprintf("Arrow %s decoded %d paired historical rows\n", packageVersion("arrow"), nrow(actual)))
