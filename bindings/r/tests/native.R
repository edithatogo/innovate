library(innovate.R)
payload <- list(inputs=list(time=c(0,1,2)), parameters=list(L=100,k=1,x0=1))
x <- kernel_predict_model("logistic", payload)
stopifnot(x$result$values[[2]] == 50, x$metadata$runtime == "rust_native")
stopifnot(length(kernel_discover_models()$models) == 5)
error <- tryCatch(kernel_predict_model("latent_process", payload), error=function(e) conditionMessage(e))
stopifnot(is.character(error), grepl("unsupported_native_operation", error))
for (i in seq_len(100)) stopifnot(kernel_predict_model("logistic", payload)$result$values[[2]] == 50)
cat("R native conformance passed\n")

stopifnot(x$schema_version == "2.0")
bass <- function(t) { d <- exp(-0.38*t); 100*(1-d)/(1+0.35/0.03*d) }
models <- list(
  logistic=list(parameters=list(L=100,k=0.6,x0=8), expected=100/(1+exp(-0.6*(3-8)))),
  gompertz=list(parameters=list(a=100,b=3,c=0.3), expected=100*exp(-3*exp(-0.3*3))),
  fisher_pry=list(parameters=list(alpha=0.6,t0=8), expected=1/(1+exp(-0.6*(3-8)))),
  bass=list(parameters=list(p=0.03,q=0.35,m=100), expected=bass(3)),
  norton_bass=list(parameters=list(p1=0.03,q1=0.35,m1=100), expected=bass(3))
)
for (name in names(models)) {
  model <- models[[name]]
  r <- kernel_predict_model(name, list(inputs=list(time=c(3,4)), parameters=model$parameters))$result
  values <- if (!is.null(r$values)) unlist(r$values) else unlist(r$rows)
  stopifnot(abs(values[[1]] - model$expected) < 1e-10)
}
cat("R five-model analytic conformance passed\n")
