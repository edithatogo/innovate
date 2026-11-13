library(innovate.R)
library(jsonlite)

base <- Sys.getenv("SCENARIO_BINDING_INPUTS")
stopifnot(nzchar(base))
expected <- fromJSON(file.path(base, "expected.json"), simplifyVector = FALSE)
stopifnot(length(expected) == 24)
for (i in seq_along(expected)) {
  prefix <- sprintf("%02d", i - 1)
  import_request <- paste(readLines(file.path(base, paste0(prefix, "-import.json")),
                                    warn = FALSE), collapse = "")
  imported <- fromJSON(kernel_call_json(import_request, raw = TRUE), simplifyVector = FALSE)$result
  stopifnot(isTRUE(imported$replay_available))
  replay_request <- paste(readLines(file.path(base, paste0(prefix, "-replay.json")),
                                    warn = FALSE), collapse = "")
  row <- fromJSON(kernel_call_json(replay_request, raw = TRUE),
                  simplifyVector = FALSE)$result$executions[[1]]
  stopifnot(is.null(row$error))
  for (key in names(expected[[i]]$expected)) {
    comparison <- all.equal(row$result[[key]], expected[[i]]$expected[[key]],
                            tolerance = 1e-12, check.attributes = FALSE)
    if (!isTRUE(comparison)) stop(expected[[i]]$id, " ", key, " ", comparison)
  }
}
for (i in 0:4) {
  request <- paste(readLines(file.path(base, sprintf("legacy-%02d.json", i)),
                            warn = FALSE), collapse = "")
  response <- kernel_call_json(request, raw = TRUE)
  stopifnot(grepl('"seed":18446744073709551615', response, fixed = TRUE))
  stopifnot(!fromJSON(response, simplifyVector = FALSE)$result$replay_available)
}
cat(R.version.string, "matched 24 historical replays and preserved 5 legacy seed tokens.\n")
