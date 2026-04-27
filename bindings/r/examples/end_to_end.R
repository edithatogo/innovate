repo_root <- {
  resolved <- NULL
  for (start in normalizePath(c(getwd(), file.path(getwd(), "innovate")), mustWork = FALSE)) {
    candidate <- start
    repeat {
      if (file.exists(file.path(candidate, "bindings", "r", "R", "kernel_bridge.R"))) {
        resolved <- candidate
        break
      }
      parent <- dirname(candidate)
      if (identical(parent, candidate)) {
        break
      }
      candidate <- parent
    }
    if (!is.null(resolved)) {
      break
    }
  }
  if (is.null(resolved)) {
    stop("Unable to locate the innovate repository root", call. = FALSE)
  }
  resolved
}
source(file.path(repo_root, "bindings", "r", "R", "kernel_bridge.R"))

time <- c(0, 1, 2, 3, 4)
observed <- c(0.02, 0.06, 0.12, 0.25, 0.41)

discovery <- kernel_discover_models()
bass <- discovery[discovery$key == "bass", , drop = FALSE]

fit <- kernel_fit_model(
  kernel_request(
    operation = "fit_model",
    model_key = bass$key[[1]],
    payload = list(
      inputs = list(time = time, observed = observed),
      model_kwargs = list()
    )
  )
)

diagnostics <- kernel_extract_diagnostics(fit)
prediction <- kernel_predict_model(
  kernel_request(
    operation = "predict_model",
    model_key = bass$key[[1]],
    payload = list(
      inputs = list(time = time),
      state = fit$state
    )
  )
)

diagnose <- kernel_diagnose_model(
  kernel_request(
    operation = "diagnose_model",
    model_key = bass$key[[1]],
    payload = list(
      inputs = list(time = time, observed = observed),
      state = fit$state
    )
  )
)

invisible(list(
  discovery = discovery,
  fit = fit,
  diagnostics = diagnostics,
  prediction = prediction,
  diagnose = diagnose
))
