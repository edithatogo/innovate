library(innovate.R)
payload <- list(inputs=list(time=c(0,1,2)), parameters=list(L=100,k=1,x0=1))
x <- kernel_predict_model("logistic", payload)
stopifnot(x$result$values[[2]] == 50, x$metadata$runtime == "rust_native")
stopifnot(length(kernel_discover_models()$models) == 7)
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

# A one-point forecast is still a vector in the shared wire contract.
single <- kernel_predict_model("logistic", list(inputs=list(time=c(1)), parameters=list(L=100,k=1,x0=1)))
stopifnot(single$result$values[[1]] == 50)

fraction <- function(t,p,q) ifelse(t<=0,0,(1-exp(-(p+q)*t))/(1+q/p*exp(-(p+q)*t)))
nb_oracle <- function(t) {
  f1 <- fraction(t,0.03,0.2); f2 <- fraction(t-4,0.02,0.3)
  c(1000*f1*(1-f2),(1500+1000*f1)*f2)
}
nb_options <- list(n_generations=2,variant="successive_generations",launch_times=c(0,4))
nb_params <- list(p1=0.03,q1=0.2,m1=1000,p2=0.02,q2=0.3,m2=1500)
t <- seq(0,20,0.5)
observed <- lapply(t,nb_oracle)
nb <- kernel_fit_model("norton_bass", list(inputs=list(time=t,observed=observed),constructor_kwargs=nb_options))
stopifnot(isTRUE(nb$result$diagnostics$optimizer$converged), nb$result$diagnostics$identifiability$rank == 6)
future <- kernel_predict_model("norton_bass", list(inputs=list(time=c(21,25,30)), state=nb$result$state))
for (i in 1:3) stopifnot(max(abs(unlist(future$result$rows[[i]])-nb_oracle(c(21,25,30)[i])))<0.01)
stopifnot(future$result$rates[[3]][[1]] < 0)
cat("R multi-generation fit, forecast, rates and single-row transport passed\n")

# Independent analytic limits of the competitive equations (zero interactions).
competitive_models <- list(
  lotka_volterra=list(parameters=list(alpha1=0.3,beta1=0,alpha2=0.4,beta2=0),
    options=list(initial_values=c(0.05,0.02)),
    expected=c(1/(1+19*exp(-0.3*3)),1/(1+49*exp(-0.4*3)))),
  multi_product=list(parameters=list(p1=0.03,q1=0.35,m1=100,p2=0.03,q2=0.35,m2=200,alpha_1_2=0,alpha_2_1=0),
    options=list(n_products=2), expected=c(bass(3),2*bass(3)))
)
for (name in names(competitive_models)) {
  model <- competitive_models[[name]]
  result <- kernel_predict_model(name, list(inputs=list(time=c(3)),parameters=model$parameters,constructor_kwargs=model$options))$result
  stopifnot(length(result$rows[[1]])==2, max(abs(unlist(result$rows[[1]])-model$expected))<1e-6)
  stopifnot(length(result$rates[[1]])==2)
}
# Optional null fields must survive recursive array preservation.
null_options <- kernel_predict_model("logistic",list(inputs=list(time=c(1)),parameters=list(L=100,k=1,x0=1),constructor_kwargs=list(t_event=NULL)))
stopifnot(null_options$result$values[[1]]==50)
cat("R competitive analytic limits, two-output rates and optional-null transport passed\n")

extreme_fit <- tryCatch(kernel_fit_model("logistic",list(inputs=list(time=c(0,1,2,3),observed=rep(1e-12,4)),fit_options=list(initial_parameters=list(L=1e140,k=1e15,x0=1)))),error=function(e) conditionMessage(e))
stopifnot(is.character(extreme_fit),grepl("finite numerical domain",extreme_fit))
cat("R installed native overflow guard passed\n")
