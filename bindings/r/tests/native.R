fixture_dir <- if (dir.exists("fixtures")) "fixtures" else "bindings/r/tests/fixtures"
fixture_path <- function(name) file.path(fixture_dir, name)

library(innovate.R)
payload <- list(inputs=list(time=c(0,1,2)), parameters=list(L=100,k=1,x0=1))
x <- kernel_predict_model("logistic", payload)
stopifnot(x$result$values[[2]] == 50, x$metadata$runtime == "rust_native")
stopifnot(length(kernel_discover_models()$models) == 16)
error <- tryCatch(kernel_predict_model("not_a_model", payload), error=function(e) conditionMessage(e))
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
kairos_payload <- list(inputs=list(time=c(0,1,5,20)), parameters=list(p=0.2,q=0.3,m=100), simulation_options=list(engine="kairos_des",seed=42,population=100))
kairos_a <- kernel_simulate_model("bass",kairos_payload)
stopifnot(identical(kairos_a,kernel_simulate_model("bass",kairos_payload)),kairos_a$result$counts[[1]]==0,kairos_a$metadata$engine=="kairos_des")
cat("R Kairos native replay passed\n")
abm_payload <- list(inputs=list(time=c(0,1,3,5)),parameters=list(p=0.2,q=0.3,m=100),simulation_options=list(engine="kairos_abm",seed=42,population=100,covariate=list(time=c(0,3),values=c(-1,0),beta_p=0.2,beta_q=0.3)))
abm <- kernel_simulate_model("bass",abm_payload)$result
stopifnot(all(unlist(abm$counts)[1:3]==0),abm$counts[[4]]>0,length(abm$agents)==100,sum(vapply(abm$agents,function(a)a$adopted,logical(1)))==abm$counts[[4]])
cat("R ABM covariate forcing and agent conservation passed\n")
historical <- kernel_predict_model("logistic",list(inputs=list(time=c(1),covariates=list(price=c(1))),parameters=list(L=100,k=0.5,x0=2,beta_L_price=10,beta_k_price=0.1,beta_x0_price=0.2),constructor_kwargs=list(covariates=list("price"))))$result
stopifnot(abs(historical$values[[1]]-110/(1+exp(0.72)))<1e-10)
initial_payload <- list(inputs=list(time=c(0,20)),parameters=list(p=0.2,q=0.3,m=100),simulation_options=list(engine="kairos_abm",seed=42,population=100,initial_adopters=20))
stopifnot(kernel_simulate_model("bass",initial_payload)$result$counts[[1]]==20)
cat("R historical Logistic and initial ABM conformance passed\n")
network_payload <- list(inputs=list(time=c(0,5,20)),parameters=list(p=1e12,q=0,m=3),simulation_options=list(engine="kairos_network",population=3,seed=42,edges=list(),capacity=1,service_ticks=5,ticks_per_unit=1))
net <- kernel_simulate_model("bass",network_payload)$result
stopifnot(net$resource_metrics$peak_queue_length == 2, net$resource_metrics$peak_in_service == 1)
stopifnot(net$resource_metrics$idle_slot_ticks == 5, net$resource_metrics$over_capacity_slot_ticks == 0)
stopifnot(all(unlist(net$counts)==c(0,0,3)),all(vapply(net$completions,function(e)e$tick,numeric(1))==c(6,11,16)))
closure_payload <- network_payload
closure_payload$simulation_options$schedule <- list(list(tick=3,capacity=0))
closed <- kernel_simulate_model("bass",closure_payload)$result
stopifnot(closed$resource_metrics$idle_slot_ticks == 1,
          closed$resource_metrics$over_capacity_slot_ticks == 3,
          length(closed$completions) == 1, closed$completions[[1]]$tick == 6)
wait_payload <- network_payload
wait_payload$simulation_options$max_wait_ticks <- 4
expired <- kernel_simulate_model("bass",wait_payload)$result
stopifnot(all(unlist(expired$counts)==c(0,0,2)),length(expired$abandonments)==1,expired$abandonments[[1]]$tick==7)
cat("R Kairos waiting deadline passed\n")
cg <- kernel_predict_model("complementary_goods",list(inputs=list(time=c(0,3)),parameters=list(k1=0.3,k2=0.4,c1=0,c2=0),constructor_kwargs=list(initial_values=c(0.05,0.02))))$result
stopifnot(abs(cg$rows[[2]][[1]]-1/(1+19*exp(-0.9)))<1e-8)
cat("R complementary goods and network FIFO conformance passed\n")
cg_time <- 0:25
cg_capacity <- 0.3/0.27
cg_y <- lapply(cg_time,function(t)rep(cg_capacity/(1+(cg_capacity/0.05-1)*exp(-0.3*t)),2))
cg_fit <- kernel_fit_model("complementary_goods",list(inputs=list(time=cg_time,observed=cg_y),fit_options=list(initial_parameters=list(k1=0.28,k2=0.28,c1=0.025,c2=0.025))))$result
stopifnot(cg_fit$diagnostics$optimizer$converged,abs(cg_fit$parameters$c1-0.03)<1e-4)
cat("R complementary analytic parameter recovery passed\n")
network_payload$parameters$p <- 0
network_payload$parameters$q <- 3
network_payload$simulation_options$edges <- list(c(0,1),c(1,2))
network_payload$simulation_options$initial_agents <- c(0)
net_seeded <- kernel_simulate_model("bass",network_payload)$result
stopifnot(net_seeded$counts[[1]]==1,net_seeded$counts[[3]]==3)
cat("R singleton initial-node and pure contagion passed\n")
lock_params <- list(alpha1=0.3,alpha2=0.2,beta1=0,beta2=0,gamma1=0,gamma2=0,m=1000)
lock_result <- kernel_predict_model("lock_in",list(inputs=list(time=c(10,20)),parameters=lock_params,constructor_kwargs=list(initial_values=c(10,0))))$result
stopifnot(abs(lock_result$rows[[1]][[1]]-1000/(1+99*exp(-3)))<1e-5,lock_result$rows[[1]][[2]]==0)
cat("R lock-in analytic forecast passed\n")
network_curve <- kernel_predict_model("network_diffusion",list(inputs=list(time=c(0,3)),constructor_kwargs=list(base_model="logistic",adjacency=list(c(0,2),c(0,0)),node_labels=list("a","b")),parameters=list(spillover_strength=0.2,node_0_L=100,node_0_k=1,node_0_x0=3,node_1_L=200,node_1_k=1,node_1_x0=3)))$result
stopifnot(abs(network_curve$rows[[2]][[1]]-70)<1e-10,network_curve$rows[[2]][[2]]==100)
cat("R historical weighted network prediction passed\n")
policy_payload <- list(inputs=list(time=c(0,2,3)),parameters=list(decay=0.5,base_L=100,base_k=1,base_x0=2),constructor_kwargs=list(base_model="logistic",event_times=c(2),event_effects=c(0.5),event_labels=c("policy")))
policy_prediction <- kernel_predict_model("policy_hazard",policy_payload)$result
stopifnot(policy_prediction$values[[2]]==75)
cat("R singleton policy event boundary passed\n")
p <- list(inputs=list(time=c(0,100)),parameters=list(p=1e12,q=0,m=3),simulation_options=list(engine="kairos_network",population=3,seed=42,edges=list(),capacity=1,service_ticks=5,ticks_per_unit=1,innovation_multipliers=c(1,0,1),agent_service_ticks=c(4,9,7)))
a <- kernel_simulate_model("bass",p)$result
stopifnot(identical(unlist(a$counts),c(0L,2L)))
expected <- 1
for (i in seq_along(a$requests)) {
  stopifnot(a$requests[[i]]$agent == a$completions[[i]]$agent)
  expected <- expected+c(4,9,7)[a$requests[[i]]$agent+1]
  stopifnot(a$completions[[i]]$tick == expected)
}
p$simulation_options$population <- 1
p$simulation_options$innovation_multipliers <- 1
p$simulation_options$imitation_multipliers <- 0
p$simulation_options$agent_service_ticks <- 4
p$parameters$m <- 1
stopifnot(kernel_simulate_model("bass",p)$result$counts[[2]] == 1)
cat("R heterogeneous service and singleton transport passed\n")
p <- list(inputs=list(time=c(0,2,4)),parameters=list(base_L=100,base_k=1,base_x0=2,latent_smoothing=0.3,latent_noise=0),constructor_kwargs=list(base_model="logistic",latent_time=c(0,4),latent_values=c(2,6)))
stopifnot(kernel_predict_model("latent_process",p)$result$values[[2]]==54)
p$constructor_kwargs$latent_time <- 0
p$constructor_kwargs$latent_values <- 2
stopifnot(kernel_predict_model("latent_process",p)$result$values[[2]]==52)
time <- 0:20
y <- 100/(1+exp(-0.5*(time-8)))+2+0.1*sin(time)
f <- kernel_fit_model("latent_process",list(inputs=list(time=time,observed=y),constructor_kwargs=list(base_model="logistic",smoothing=0.4)))$result
stopifnot(length(f$state$constructor_kwargs$latent_values)==21)
q <- f$parameters
expected <- q$base_L/(1+exp(-q$base_k*(30-q$base_x0)))+tail(unlist(f$state$constructor_kwargs$latent_values),1)
stopifnot(abs(kernel_predict_model("latent_process",list(inputs=list(time=30),state=f$state))$result$values[[1]]-expected)<1e-10)
cat("R latent fit, saved state and singleton arrays passed\n")
time <- 0:29
y <- ifelse(time<15,100/(1+exp(-0.4*(time-6))),200/(1+exp(-0.3*(time-18))))
f <- kernel_fit_model("regime_switching",list(inputs=list(time=time,observed=y),constructor_kwargs=list(base_model="logistic",prediction_mode="fixed_time",split_index=15)))$result
stopifnot(f$state$constructor_kwargs$split_time==15,abs(f$parameters$regime_1_k-0.3)<0.001)
a <- kernel_predict_model("regime_switching",list(inputs=list(time=c(20,30,40)),state=f$state))$result
b <- kernel_predict_model("regime_switching",list(inputs=list(time=40),state=f$state))$result
stopifnot(a$values[[3]]==b$values[[1]])
cat("R fixed-time regime fit and forecast state passed\n")
f <- kernel_fit_model("regime_switching",list(inputs=list(time=c(0,1),observed=c(0.11920292202211755,0.2689414213699951)),constructor_kwargs=list(base_model="fisher_pry")))$result
stopifnot(f$details$regime_count==1,abs(kernel_predict_model("regime_switching",list(inputs=list(time=2),state=f$state))$result$values[[1]]-0.5)<1e-8)
cat("R short Fisher-Pry regime and singleton detected boundary passed\n")
time <- 0:24
y <- list(a=100/(1+exp(-0.4*(time-7))),b=200/(1+exp(-0.3*(time-9))))
f <- kernel_fit_model("hierarchical",list(inputs=list(time=time,observed=y),constructor_kwargs=list(base_model="logistic",groups=c("a","b"),output="groups")))$result
stopifnot(abs(f$parameters$global_L-150)<0.01)
a <- kernel_predict_model("hierarchical",list(inputs=list(time=30),state=f$state))$result
stopifnot(abs(a$rows[[1]][[1]]-100/(1+exp(-9.2)))<0.01,abs(a$rows[[1]][[2]]-200/(1+exp(-6.3)))<0.01)
p <- list(inputs=list(time=2,observed=list(a=50)),constructor_kwargs=list(base_model="logistic",groups="a"),parameters=list(global_L=100,global_k=1,global_x0=2,a_L=0,a_k=0,a_x0=0))
stopifnot(kernel_diagnose_model("hierarchical",p)$result$diagnostics$metrics$RSS==0)
cat("R hierarchical grouped state and singleton mapping passed\n")
time <- 0:24
y <- 100/(1+exp(-0.4*(time-7)));y[21] <- 300
weights <- rep(1,25);weights[21] <- 0
f <- kernel_fit_model("logistic",list(inputs=list(time=time,observed=y),fit_options=list(weights=weights,initial_parameters=list(L=110,k=0.35,x0=8))))$result
stopifnot(abs(f$parameters$L-100)<1e-5)
time <- seq(0,24.5,0.5)
y <- ifelse((seq_along(time)-1)%%2==0,100/(1+exp(-0.4*(time-6))),200/(1+exp(-0.3*(time-10))))
parameters <- list(model_0_L=100,model_0_k=0.4,model_0_x0=6,model_1_L=200,model_1_k=0.3,model_1_x0=10,weight_0=0.4,weight_1=0.6,noise_scale=2)
f <- kernel_fit_model("mixture",list(inputs=list(time=time,observed=y),constructor_kwargs=list(components=c("logistic","logistic"),max_iter=30,noise_floor=0.001),fit_options=list(initial_parameters=parameters)))$result
stopifnot(f$optimizer$converged,abs(f$parameters$model_1_L-200)<0.05,kernel_predict_model("mixture",list(inputs=list(time=40),state=f$state))$result$values[[1]]>140)
p <- list(inputs=list(time=2),constructor_kwargs=list(components="logistic",weights=1,mode="historical_adoption"),parameters=list(model_0_L=100,model_0_k=1,model_0_x0=2,weight_0=1))
stopifnot(kernel_predict_model("mixture",p)$result$values[[1]]==50)
cat("R weighted fit, Gaussian mixture and singleton component passed\n")
p <- list(inputs=list(time=c(0,3,10,20)),constructor_kwargs=list(components=c("bass","logistic"),initial_values=c(0,10)),parameters=list(p_1=0.03,q_1=0.35,m_1=100,L_2=200,k_2=0.4,x0_2=99,alpha_1_2=0,alpha_2_1=0))
a <- kernel_predict_model("composite",p)$result
logistic <- 200/(1+19*exp(-4))
stopifnot(abs(a$rows[[3]][[2]]-logistic)<1e-6,abs(a$rates[[3]][[2]]-0.4*logistic*(1-logistic/200))<1e-6)
time <- 0:24
y <- 100*(1-exp(-0.38*time))/(1+0.35/0.03*exp(-0.38*time))
f <- kernel_fit_model("composite",list(inputs=list(time=time,observed=y),constructor_kwargs=list(components="bass"),fit_options=list(initial_parameters=list(p_1=0.025,q_1=0.32,m_1=110))))$result
stopifnot(f$diagnostics$optimizer$converged,kernel_predict_model("composite",list(inputs=list(time=30),state=f$state))$result$rows[[1]][[1]]>99.5)
cat("R composite matrix rates and singleton fitted origin passed\n")
p <- list(inputs=list(time=c(0,1,2,4),covariates=list(dose=c(0,1,2,4))),constructor_kwargs=list(covariates="dose"),parameters=list(p=0.1,q=0,m=100,beta_p_dose=0.02,beta_q_dose=0,beta_m_dose=0))
a <- kernel_predict_model("bass",p)$result
stopifnot(abs(a$values[[4]]-100*(1-exp(-0.56)))<1e-7)
a <- kernel_predict_model("bass",list(inputs=list(time=c(5,6),covariates=list(dose=c(5,6))),state=a$state))$result
stopifnot(abs(a$values[[2]]-100*(1-exp(-0.96)))<1e-7)
p <- list(inputs=list(time=1),constructor_kwargs=list(covariates="dose",covariate_history=list(time=0,values=list(dose=0))),parameters=list(p=0.1,q=0,m=100,beta_p_dose=0.02,beta_q_dose=0,beta_m_dose=0))
stopifnot(abs(kernel_predict_model("bass",p)$result$values[[1]]-100*(1-exp(-0.1)))<1e-7)
time <- seq(0,8,0.5)
boundary <- 100*exp(log(0.01)*exp(-0.6))
y <- ifelse(time<2,100*exp(log(0.01)*exp(-0.3*time)),200*exp(log(boundary/200)*exp(-0.2*(time-2))))
f <- kernel_fit_model("gompertz",list(inputs=list(time=time,observed=y),constructor_kwargs=list(t_event=2,initial_value=1),fit_options=list(initial_parameters=list(a=100,b=77,c=0.3,a_post=200,b_post=99,c_post=0.18))))$result
stopifnot(f$diagnostics$optimizer$converged,abs(kernel_predict_model("gompertz",list(inputs=list(time=10),state=f$state))$result$values[[1]]-200*exp(log(boundary/200)*exp(-1.6)))<0.001)
cat("R mechanistic covariates, events, history and fit passed\n")
time <- seq(0,5,0.5);dose <- 1+sin(0.7*time)
p <- list(inputs=list(time=time,covariates=list(dose=dose)),constructor_kwargs=list(covariates="dose"),parameters=list(p=0.1,q=0,m=100,beta_p_dose=0.02,beta_q_dose=0,beta_m_dose=0))
a <- kernel_predict_model("bass",p)$result
b <- kernel_predict_model("bass",list(inputs=p$inputs,state=a$state))$result
stopifnot(max(abs(unlist(a$values)-unlist(b$values)))<1e-7)
cat("R nontrivial forcing history replays without revision\n")

weighted_network <- list(inputs=list(time=c(0,5,20)),parameters=list(p=0,q=1e12,m=3),simulation_options=list(engine="kairos_network",population=3,seed=42,directed=TRUE,edges=list(c(0,1)),edge_weights=1e308,initial_agents=0,capacity=1,service_ticks=5,ticks_per_unit=1))
weighted_network$simulation_options$schedule <- list(list(tick=6,edges=list(c(1,2)),service_ticks=2,capacity=2,p=1e12,q=0))
weighted_network$simulation_options$agent_priorities <- c(-2,0,3)
weighted_result <- kernel_simulate_model("bass",weighted_network)$result
stopifnot(all(unlist(weighted_result$counts)==c(1,1,3)),all(vapply(weighted_result$completions,function(e)e$tick,numeric(1))==c(6,9)))
stopifnot(all(unlist(weighted_result$metadata$agent_priorities)==c(-2,0,3)),weighted_result$metadata$current_capacity==2,weighted_result$metadata$current_p==1e12,weighted_result$metadata$current_q==0)
cat("R directed weighted Kairos resource and priorities passed\n")

retry_options <- list(engine="kairos_network",population=2,seed=42,edges=list(),capacity=1,service_ticks=5,ticks_per_unit=1,max_queue_length=0,retry=list(delay_ticks=4,max_retries=1))
retry_payload <- list(inputs=list(time=c(0,6,12)),parameters=list(p=1e12,q=0,m=2),simulation_options=retry_options)
retry_result <- kernel_simulate_model("bass",retry_payload)$result
stopifnot(all(unlist(retry_result$counts)==c(0,1,2)),length(retry_result$balks)==1,retry_result$balks[[1]]$tick==2,length(retry_result$retry_eligibility)==1,retry_result$retry_eligibility[[1]]$tick==6)
cat("R Kairos queue limit and retry passed\n")

nb_parameters <- list(p1=0.1,q1=0,m1=100,p2=0.1,q2=0,m2=100,beta_p1_price=0.05,beta_q1_price=0,beta_m1_price=0,beta_p2_price=0.05,beta_q2_price=0,beta_m2_price=0)
nb_payload <- list(inputs=list(time=c(0,1,2),covariates=list(price=c(0,1,2))),parameters=nb_parameters,constructor_kwargs=list(n_generations=2,covariates="price"))
nb_result <- kernel_predict_model("norton_bass",nb_payload)$result
nb_forecast <- kernel_predict_model("norton_bass",list(inputs=list(time=3),state=nb_result$state))$result
stopifnot(abs(nb_result$rows[[3]][[1]]-30*exp(-0.3))<1e-5,abs(nb_forecast$rows[[1]][[1]]-50*exp(-0.5))<1e-5)
cat("R Norton Bass forcing history passed\n")

lv_fixture <- jsonlite::fromJSON(fixture_path("lotka_forcing_scipy.json"),simplifyVector=FALSE)
lv_start <- lv_fixture$parameters
lv_start$alpha1 <- 0.29
lv_start$beta_beta1_price <- -0.025
for (lv_options in list(list(initial_parameters=lv_start,max_iterations=150),list(max_iterations=150))) {
  lv_payload <- list(inputs=list(time=unlist(lv_fixture$time),observed=lv_fixture$rows,covariates=list(price=unlist(lv_fixture$covariates$price))),constructor_kwargs=list(initial_values=unlist(lv_fixture$initial),covariates="price"),fit_options=lv_options)
  lv_fit <- kernel_fit_model("lotka_volterra",lv_payload)$result
  lv_forecast <- kernel_predict_model("lotka_volterra",list(inputs=list(time=unlist(lv_fixture$forecast_time)),state=lv_fit$state))$result
  stopifnot(lv_fit$diagnostics$optimizer$converged,max(abs(unlist(lv_forecast$rows)-unlist(lv_fixture$forecast_rows)))<1e-5)
}
cat("R interacting Lotka forcing fit and history passed\n")

mp_fixture <- jsonlite::fromJSON(
  fixture_path("multi_forcing_scipy.json"), simplifyVector=FALSE
)
for (layout in c("corrected", "historical")) {
  mp_payload <- list(
    inputs=list(time=unlist(mp_fixture$time), covariates=list(price=unlist(mp_fixture$covariates$price))),
    parameters=mp_fixture$parameters,
    constructor_kwargs=list(n_products=2, covariates="price", coefficient_layout=layout)
  )
  mp_result <- kernel_predict_model("multi_product", mp_payload)$result
  stopifnot(max(abs(unlist(tail(mp_result$rows, 1))-unlist(tail(mp_fixture$cases[[layout]]$rows, 1))))<1e-5)
  mp_forecast <- kernel_predict_model(
    "multi_product", list(inputs=list(time=unlist(mp_fixture$forecast_time)), state=mp_result$state)
  )$result
  stopifnot(max(abs(unlist(mp_forecast$rows)-unlist(mp_fixture$cases[[layout]]$forecast_rows)))<1e-5)
}
mp_start <- mp_fixture$parameters
mp_start$p1 <- 0.0305
mp_fit <- kernel_fit_model("multi_product", list(
  inputs=list(time=unlist(mp_fixture$time), covariates=list(price=unlist(mp_fixture$covariates$price)),
              observed=mp_fixture$cases$corrected$rows),
  constructor_kwargs=list(n_products=2, covariates="price"),
  fit_options=list(initial_parameters=mp_start, max_iterations=200)
))$result
mp_forecast <- kernel_predict_model(
  "multi_product", list(inputs=list(time=unlist(mp_fixture$forecast_time)), state=mp_fit$state)
)$result
stopifnot(mp_fit$diagnostics$optimizer$converged,
          max(abs(unlist(mp_forecast$rows)-unlist(mp_fixture$cases$corrected$forecast_rows)))<1e-3)
cat("R multi product forcing layouts, fit and forecast passed\n")

cg_fixture <- jsonlite::fromJSON(
  fixture_path("complementary_forcing_scipy.json"), simplifyVector=FALSE
)
cg_start <- cg_fixture$parameters
cg_start$k1 <- 0.29
cg_start$beta_c2_price <- -0.007
for (cg_options in list(list(initial_parameters=cg_start,max_iterations=150),list(max_iterations=150))) {
  cg_fit <- kernel_fit_model("complementary_goods", list(
    inputs=list(time=unlist(cg_fixture$time), covariates=list(price=unlist(cg_fixture$covariates$price)),
                observed=cg_fixture$rows),
    constructor_kwargs=list(initial_values=unlist(cg_fixture$initial), covariates="price"),
    fit_options=cg_options
  ))$result
  cg_forecast <- kernel_predict_model(
    "complementary_goods", list(inputs=list(time=unlist(cg_fixture$forecast_time)), state=cg_fit$state)
  )$result
  stopifnot(cg_fit$diagnostics$optimizer$converged,
            max(abs(unlist(cg_forecast$rows)-unlist(cg_fixture$forecast_rows)))<1e-5)
}
cat("R complementary goods forcing fit and history passed\n")

li_fixture <- jsonlite::fromJSON(
  fixture_path("lock_forcing_scipy.json"), simplifyVector=FALSE
)
li_start <- li_fixture$parameters
li_start$alpha1 <- 0.29
li_fit <- kernel_fit_model("lock_in", list(
  inputs=list(time=unlist(li_fixture$time), covariates=list(price=unlist(li_fixture$covariates$price)),
              observed=li_fixture$rows),
  constructor_kwargs=list(initial_values=unlist(li_fixture$initial), covariates="price"),
  fit_options=list(initial_parameters=li_start, max_iterations=200)
))$result
li_forecast <- kernel_predict_model(
  "lock_in", list(inputs=list(time=unlist(li_fixture$forecast_time)), state=li_fit$state)
)$result
stopifnot(li_fit$diagnostics$optimizer$converged,
          li_fit$diagnostics$structural_identifiability$parameters==14,
          max(abs(unlist(li_forecast$rows)-unlist(li_fixture$forecast_rows)))<1e-3)
li_parameters <- list(alpha1=0,alpha2=0,beta1=0,beta2=0,gamma1=0,gamma2=0,m=100)
for (name in names(li_parameters)) li_parameters[[paste0("beta_",name,"_price")]] <- 0
li_parameters$beta_m_price <- -10
li_projected <- kernel_predict_model("lock_in", list(
  inputs=list(time=c(0,1,2,3),covariates=list(price=c(0,1,2,0))),
  parameters=li_parameters,constructor_kwargs=list(initial_values=c(95,0),covariates="price")
))$result
stopifnot(all(vapply(li_projected$rows,function(row)row[[1]],numeric(1))==c(95,90,80,95)))
cat("R lock in forcing fit, forecast and capacity projection passed\n")

event_fixture <- jsonlite::fromJSON(
  fixture_path("ecosystem_events_scipy.json"), simplifyVector=FALSE
)
for (model in c("lotka_volterra", "complementary_goods", "lock_in", "multi_product")) {
  event_case <- event_fixture$cases[[model]]
  event_result <- kernel_predict_model(model, list(
    inputs=list(time=unlist(event_fixture$time),covariates=list(price=unlist(event_fixture$covariates$price))),
    parameters=event_case$parameters,constructor_kwargs=event_case$constructor_kwargs
  ))$result
  stopifnot(max(abs(unlist(tail(event_result$rows,1))-unlist(tail(event_case$rows,1))))<1e-5)
  event_future <- kernel_predict_model(
    model,list(inputs=list(time=unlist(event_fixture$forecast_time)),state=event_result$state)
  )$result
  stopifnot(max(abs(unlist(event_future$rows)-unlist(event_case$forecast_rows)))<1e-5)
}
event_case <- event_fixture$cases$lotka_volterra
event_start <- event_case$parameters
event_start$alpha1 <- 0.29
event_fit <- kernel_fit_model("lotka_volterra", list(
  inputs=list(time=unlist(event_fixture$time),covariates=list(price=unlist(event_fixture$covariates$price)),
              observed=event_case$rows),
  constructor_kwargs=event_case$constructor_kwargs,
  fit_options=list(initial_parameters=event_start,max_iterations=200)
))$result
stopifnot(event_fit$diagnostics$optimizer$converged,!event_fit$diagnostics$parameter_events$parameters_estimated)
single_event <- kernel_predict_model("lotka_volterra", list(
  inputs=list(time=c(0,2)),parameters=list(alpha1=0.2,beta1=0,alpha2=0.1,beta2=0),
  constructor_kwargs=list(initial_values=c(0.1,0.2),
    parameter_events=list(list(time=1,parameters=list(alpha1=0.4))))
))$result
stopifnot(abs(single_event$rows[[2]][[1]]-1/(1+9*exp(-0.6)))<1e-8)
cat("R ecosystem parameter events, fit and singleton scalar time passed\n")

for (fixture_name in c("norton_events_scipy.json","successive_forcing_scipy.json")) for (explicit_start in c(TRUE,FALSE)) {
  nb_event_fixture <- jsonlite::fromJSON(
    fixture_path(fixture_name), simplifyVector=FALSE
  )
  nb_fit_options <- list(max_iterations=200)
  if (explicit_start) {
    nb_fit_options$initial_parameters <- nb_event_fixture$parameters
    nb_fit_options$initial_parameters$p1 <- 0.0305
  }
  nb_event_fit <- kernel_fit_model("norton_bass", list(
    inputs=list(time=unlist(nb_event_fixture$time),covariates=list(price=unlist(nb_event_fixture$covariates$price)),
                observed=nb_event_fixture$rows),
    constructor_kwargs=nb_event_fixture$constructor_kwargs,
    fit_options=nb_fit_options
  ))$result
  nb_event_forecast <- kernel_predict_model(
    "norton_bass",list(inputs=list(time=unlist(nb_event_fixture$forecast_time)),state=nb_event_fit$state)
  )$result
  stopifnot(nb_event_fit$diagnostics$optimizer$converged,
            !nb_event_fit$diagnostics$parameter_events$parameters_estimated,
            max(abs(unlist(nb_event_forecast$rows)-unlist(nb_event_fixture$forecast_rows)))<1e-3)
}
cat("R Norton event variants, fit and forecast passed\n")

threshold_summary <- kernel_summarize_model("norton_bass", list(
  inputs=list(time=4),parameters=list(p1=0.1,q1=0.9,m1=100),
  constructor_kwargs=list(variant="successive_generations",launch_times=4)
))$result$threshold_diagnostics$generations[[1]]
stopifnot(threshold_summary$reference_time==4,
          abs(threshold_summary$half_life-log(9))<1e-12,
          abs(threshold_summary$saturation-2*log(9))<1e-12)
cat("R Norton baseline threshold summary passed\n")

forced_summary <- kernel_summarize_model("norton_bass", list(
  inputs=list(time=c(0,1,2,3)),parameters=list(p1=0.1,q1=0,m1=100),
  constructor_kwargs=list(variant="successive_generations",
    parameter_events=list(list(time=2,parameters=list(p1=0.5))))
))$result$threshold_diagnostics
peak_summary <- forced_summary$trajectory
peak <- peak_summary$generations[[1]]
stopifnot(all(unlist(peak_summary$window)==c(0,3)),peak$peak_time==2,
          abs(peak$peak_rate-50*exp(-0.2))<1e-6)
milestones <- forced_summary$intrinsic_adoption_milestones$generations[[1]]
stopifnot(abs(milestones$median-(2+(log(2)-0.2)/0.5))<1e-7,
          is.null(milestones$saturation))
cat("R Norton forced rate peak and milestones passed\n")

historical <- kernel_summarize_model("norton_bass", list(
  inputs=list(time=c(1,2,3)),parameters=list(p1=0.1,q1=0,m1=100),
  constructor_kwargs=list(initial_time=1,initial_values=12,
    parameter_events=list(list(time=2,parameters=list(m1=1000))))
))$result$threshold_diagnostics$net_level_milestones$generations[[1]]
stopifnot(historical$reference_market==100,historical$takeoff==1,
          historical$median==3,historical$saturation==3)
cat("R Norton historical level milestones passed\n")

scheduled_agents <- kernel_simulate_model("bass", list(
  inputs=list(time=c(0,2,5,8)),parameters=list(p=1e12,q=0,m=2),
  simulation_options=list(engine="kairos_network",population=2,seed=42,edges=list(),
    capacity=1,service_ticks=4,ticks_per_unit=1,innovation_multipliers=c(1,0),
    schedule=list(list(tick=2,innovation_multipliers=c(0,1))))
))$result
stopifnot(all(vapply(scheduled_agents$requests,function(x) x$tick,numeric(1))==c(1,3)),
          all(vapply(scheduled_agents$requests,function(x) x$agent,numeric(1))==c(0,1)),
          scheduled_agents$completions[[1]]$tick==5,scheduled_agents$completions[[1]]$agent==0)
cat("R scheduled Kairos agent hazards passed\n")

continuous_hazard <- kernel_simulate_model("bass", list(
  inputs=list(time=c(0,2,3,4)),parameters=list(p=1e12,q=0,m=1),
  simulation_options=list(engine="kairos_network",population=1,seed=42,edges=list(),
    capacity=1,service_ticks=1,ticks_per_unit=1,
    hazard_forcing=list(time=c(0,2,3),innovation=c(0,0,1),imitation=c(1,1,1)))
))$result
stopifnot(length(continuous_hazard$requests)==1,continuous_hazard$requests[[1]]$tick==3,
          continuous_hazard$requests[[1]]$agent==0,
          all(unlist(continuous_hazard$counts)==c(0,0,0,1)))
cat("R continuous Kairos hazard wakeup passed\n")

rolling <- kernel_rolling_average(c(1e100, 1, 3), 2L)
stopifnot(is.null(rolling[[1]]), abs(rolling[[2]] / 5e99 - 1) < 1e-14, rolling[[3]] == 2)
cat("R compensated rolling average passed\n")

legacy_request <- paste0(
  '{"schema_version":"2.0","operation":"import_scenario_artifact","payload":{"artifact":',
  '{"scenario":{"scenario_type":"baseline","name":"legacy","description":"","time_horizon":5,',
  '"time_unit":"year","reference_year":2026,"market_size":100,"initial_adoption":0},',
  '"seed":18446744073709551615,"model_type":"bass","version":"0.6.0","notes":null,',
  '"timestamp":"2026-01-01T00:00:00+00:00"}}}'
)
legacy_raw <- kernel_call_json(legacy_request, raw = TRUE)
stopifnot(is.character(legacy_raw), grepl('"seed":18446744073709551615', legacy_raw, fixed = TRUE))
stopifnot(!kernel_call_json(legacy_request)$result$replay_available)
legacy_invalid <- sub("2026-01-01T00:00:00+00:00", "bad", legacy_request, fixed = TRUE)
legacy_error <- tryCatch(kernel_call_json(legacy_invalid, raw = TRUE), error = conditionMessage)
stopifnot(grepl("invalid_request", legacy_error, fixed = TRUE))

dataset <- list(kind="adoption", time=c(0,1), adoption=c(1,2),
  provenance=list(source="synthetic", license="CC-BY-4.0",
    extraction_time="2026-11-07T00:00:00Z"))
artifact <- kernel_call("ingest_dataset", NULL, list(dataset=dataset))$result
stopifnot(isTRUE(artifact$validation$ok), nchar(artifact$dataset$provenance$checksum) == 64)
stopifnot(identical(kernel_call("ingest_dataset", payload=list(artifact=artifact))$result, artifact))
modified <- artifact
modified$dataset$adoption[[2]] <- 3
checksum_error <- tryCatch({kernel_call("ingest_dataset", payload=list(artifact=modified)); ""},
  error=conditionMessage)
stopifnot(grepl("checksum does not match", checksum_error, fixed=TRUE))
dataset$adoption <- c(1,0)
diagnostic <- tryCatch(kernel_call("ingest_dataset", NULL, list(dataset=dataset)),
  innovate_dataset_validation_error=identity)
stopifnot(inherits(diagnostic, "innovate_dataset_validation_error"),
  identical(diagnostic$code, "invalid_request"),
  identical(unlist(diagnostic$validation$failed), "monotonicity:adoption"))

timestamps <- c("2026-11-08T10:00:00+10:00", "2026-11-01T00:00:00Z", "2026-11-07T23:59:59.999Z")
weekly <- kernel_aggregate_dates(timestamps, c(4,1,2), "W")
stopifnot(identical(unlist(weekly$dates), c("2026-11-01T00:00:00Z", "2026-11-08T00:00:00Z")),
  identical(as.numeric(unlist(weekly$values)), c(3,4)))

manifest <- list(name="binding-fixture-\u6e2c\u8a66", version="1", entrypoint="demo:register",
  stability="beta", extension_points=list("diagnostics"), summary="\u0394 caf\u00e9")
registered <- kernel_call("register_extension_manifest", NULL, list(manifest=manifest))$result$manifest
stopifnot(identical(registered$stability, "provisional"),
  identical(registered$name, manifest$name), identical(registered$summary, manifest$summary),
  identical(kernel_call("list_extension_manifests")$result$manifests, list(registered)))

paired_parameters <- list(
  list(name="p",unit="period^-1",value=0.03),
  list(name="q",unit="period^-1",value=0.3),
  list(name="m",unit="people",value=100))
paired_draw <- list(draw_id=1,parameters=paired_parameters,time=c(0,1),adoption=c(0,1))
paired <- list(strategies=lapply(c("control","policy"), function(id) {
  list(scenario_id=id,time_unit="period",provenance="synthetic",draws=list(paired_draw))
}))
stopifnot(rawToChar(kernel_paired_outcomes_ipc(paired)[1:6]) == "ARROW1")
paired$strategies[[1]]$draws[[1]] <- unname(paired_draw)
paired_error <- tryCatch({kernel_paired_outcomes_ipc(paired); ""}, error=conditionMessage)
stopifnot(grepl("must be objects", paired_error, fixed=TRUE))

resource_payload <- paste(readLines(fixture_path("kairos_resource_comparisons.json")), collapse="\n")
resource_request <- paste0('{"schema_version":"2.0","operation":"execute_scenario_grid","payload":',
                           resource_payload, "}")
resource_result <- kernel_call_json(resource_request)$result
resource_rows <- resource_result$comparisons
resource_totals <- resource_rows[[1]]$result$resource_differences
stopifnot(length(resource_result$executions) == 3, resource_totals$configured_slot_ticks == -3,
          resource_totals$peak_in_service == 1,
          resource_totals$horizon_ticks == 11, resource_totals$ticks_per_unit == 1,
          resource_rows[[2]]$error$code == "invalid_comparison", is.null(resource_rows[[2]]$result))

shortened <- jsonlite::fromJSON(resource_payload, simplifyVector=FALSE)
for (i in seq_along(shortened$scenarios)) {
  shortened$scenarios[[i]]$simulation$inputs$time <- c(0, 7)
}
short_totals <- kernel_call("execute_scenario_grid", payload=shortened)$result$comparisons[[1]]$result$resource_differences
stopifnot(short_totals$waiting_at_horizon == 1, short_totals$in_service_at_horizon == 2)

edge_dataset <- list(kind="network_edges", source=c("a","a"), target=c("a","b"), weight=c(1,0),
  provenance=list(source="synthetic", license="CC-BY-4.0", extraction_time="2026-11-07T00:00:00Z"))
edge_response <- kernel_call("ingest_dataset", payload=list(dataset=edge_dataset))
stopifnot(identical(as.numeric(unlist(edge_response$result$dataset$weight)), c(1,0)),
  identical(unlist(edge_response$result$validation$warnings),
            c("self_loops:kairos_network", "zero_weights:kairos_network")))

for (estimand in c("ate", "att")) {
  estimate <- if (estimand == "ate") 6 else 7.25
  weight <- if (estimand == "ate") 3/7 else 1/4
  stratified <- list(method="stratified_treatment_contrast", outcomes=c(0,0,2,10,19,19,19),
    treatment=c(0,0,1,0,1,1,1), strata=rep(c("low","high"), c(3,4)))
  stratified$estimand <- estimand
  contrast <- kernel_call("summarize_causal", payload=stratified)$result
  stopifnot(isTRUE(all.equal(contrast$estimate, estimate)),
    isTRUE(all.equal(contrast$by_stratum$low$weight, weight)))
  stopifnot(isTRUE(all.equal(contrast$by_stratum$low$treated_fraction, 1/3)))
  if (estimand == "att") {
    with_control_only <- stratified
    with_control_only$strata[1] <- "control_only"
    overlap <- kernel_call("summarize_causal", payload=with_control_only)$result
    stopifnot(overlap$excluded_control_observations == 1, isTRUE(all.equal(overlap$estimate, estimate)))
  }
  stopifnot("conditional_standard_error" %in% names(contrast), is.null(contrast$conditional_standard_error))
  replicated <- modifyList(stratified, list(outcomes=c(0,2,4,6,10,10,12,12),
    treatment=rep(c(0,0,1,1), 2), strata=rep(c("low","high"), each=4)))
  result <- kernel_call("summarize_causal", payload=replicated)$result
  stopifnot(isTRUE(all.equal(result$conditional_standard_error, sqrt(0.5))))
  stratified$strata[7] <- "unsupported"
  contrast_error <- tryCatch({kernel_call("summarize_causal", payload=stratified); ""}, error=conditionMessage)
  stopifnot(grepl("each stratum requires treated and control observations", contrast_error, fixed=TRUE))
}

contrast_methods <- kernel_call("summarize_causal", payload=list(method="describe_treatment_contrasts"))$result$contrasts
stopifnot(identical(contrast_methods[[1]]$adjustment, "none"),
  identical(unlist(contrast_methods[[2]]$required_fields), c("outcomes", "treatment", "strata")),
  identical(unlist(contrast_methods[[2]]$estimands), c("ate", "att")))
weighted <- contrast_methods[[3]]
stopifnot(identical(unlist(weighted$required_fields), c("outcomes", "treatment", "propensities")))
weighted_payload <- list(method=weighted$method, outcomes=c(0,10,2,12),
  treatment=c(0,0,1,1), propensities=c(0.2,0.8,0.2,0.8))
weighted_result <- kernel_call("summarize_causal", payload=weighted_payload)$result
stopifnot(isTRUE(all.equal(weighted_result$estimate, -4)))
stopifnot(isTRUE(all.equal(weighted_result$fixed_propensity_standard_error, sqrt(20.48))))
singleton <- modifyList(weighted_payload, list(outcomes=c(0,2), treatment=c(0,1), propensities=c(0.5,0.5)))
singleton_result <- kernel_call("summarize_causal", payload=singleton)$result
stopifnot("fixed_propensity_standard_error" %in% names(singleton_result),
  is.null(singleton_result$fixed_propensity_standard_error))
for (arm in c("treated", "control")) {
  diagnostics <- weighted_result$weight_diagnostics[[arm]]
  stopifnot(diagnostics$observations == 2,
    isTRUE(all.equal(diagnostics$maximum_normalized_weight, 0.8)),
    isTRUE(all.equal(diagnostics$effective_sample_size, 25/17)))
}
stopifnot(identical(unlist(weighted$estimands), c("ate", "att")))
att <- kernel_call("summarize_causal", payload=c(weighted_payload, list(estimand="att")))$result
stopifnot(isTRUE(all.equal(att$estimate, -41/17)),
  att$weight_diagnostics$treated$maximum_normalized_weight == 0.5)
weighted_payload$propensities[1] <- 0
weighted_error <- tryCatch({kernel_call("summarize_causal", payload=weighted_payload); ""}, error=conditionMessage)
stopifnot(grepl("strictly between zero and one", weighted_error, fixed=TRUE))

fit_treatment <- c(0,1,0,1,0,1,1)
fit <- kernel_call("summarize_causal", payload=list(method="estimate_stratum_propensities",
  treatment=fit_treatment, strata=c("low","high","low","low","high","high","high")))$result
stopifnot(isTRUE(all.equal(unlist(fit$propensities), c(1/3,0.75,1/3,1/3,0.75,0.75,0.75))))
weighted_fit <- kernel_call("summarize_causal", payload=list(method="propensity_weighted_contrast",
  outcomes=c(0,19,0,2,10,19,19), treatment=fit_treatment, propensities=fit$propensities))$result
stopifnot(isTRUE(all.equal(weighted_fit$estimate, 6)))

matching <- contrast_methods[[4]]
stopifnot(identical(unlist(matching$estimands), "att"),
  identical(unlist(matching$required_fields), c("outcomes", "treatment", "scores", "caliper")))
matching_payload <- list(method=matching$method, outcomes=c(0,10,12,3,99),
  treatment=c(0,0,1,1,0), scores=c(0,2,2,1,0), caliper=1)
matched <- kernel_call("summarize_causal", payload=matching_payload)$result
stopifnot(isTRUE(all.equal(matched$estimate, 2.5)),
  all(vapply(matched$pairs, function(pair) as.numeric(pair$control_index), numeric(1)) == c(1,0)))
stopifnot(matched$control_diagnostics$unique_controls == 2,
  matched$control_diagnostics$effective_sample_size == 2)
reused <- kernel_call("summarize_causal",
  payload=modifyList(matching_payload, list(scores=c(0,2,2,2,0), outcome_variances=c(1,4,1,1,1))))$result
stopifnot("conditional_standard_error" %in% names(matched), is.null(matched$conditional_standard_error),
  isTRUE(all.equal(reused$conditional_standard_error, sqrt(4.5))))
stopifnot(reused$control_diagnostics$unique_controls == 1,
  reused$control_diagnostics$effective_sample_size == 1)
matching_payload$caliper <- 0.5
partial <- kernel_call("summarize_causal",
  payload=c(matching_payload, list(allow_partial=TRUE)))$result
stopifnot(identical(partial$estimand, "matched_subset_att"),
  partial$estimate == 2, identical(as.numeric(unlist(partial$unmatched_treated_indices)), 3))
matching_error <- tryCatch({kernel_call("summarize_causal", payload=matching_payload); ""}, error=conditionMessage)
stopifnot(grepl("requires a control within the caliper", matching_error, fixed=TRUE))

logistic_fit <- kernel_call("summarize_causal", payload=list(method="fit_logistic_propensities",
  treatment=c(0,1), covariates=matrix(c(-2,2), ncol=1), ridge=0.25/log(3)))$result
stopifnot(isTRUE(all.equal(unlist(logistic_fit$propensities), c(0.25,0.75), tolerance=1e-7)),
  identical(as.numeric(unlist(logistic_fit$covariate_scales)), 2))
logistic_prediction <- kernel_call("summarize_causal", payload=list(method="predict_logistic_propensities",
  fit=logistic_fit, covariates=matrix(c(0,2,-2), ncol=1)))$result
stopifnot(isTRUE(all.equal(unlist(logistic_prediction$propensities), c(0.5,0.75,0.25), tolerance=1e-7)))
logistic_score <- kernel_call("summarize_causal", payload=list(method="score_propensities",
  treatment=c(0,1,0), propensities=logistic_prediction$propensities))$result
stopifnot(isTRUE(all.equal(logistic_score$brier_score, 0.125, tolerance=1e-7)),
  isTRUE(all.equal(logistic_score$log_loss, -(log(0.5) + 2 * log(0.75)) / 3, tolerance=1e-7)))
logistic_cross_fit <- kernel_call("summarize_causal", payload=list(method="cross_fit_logistic_propensities",
  treatment=c(0,1,1,0), covariates=matrix(c(-2,1,2,-1), ncol=1),
  folds=c(9L,4L,9L,4L), ridge=0.25/log(3)))$result
low <- 1 / (1 + sqrt(3))
stopifnot(isTRUE(all.equal(unlist(logistic_cross_fit$propensities), c(0.1,1-low,0.9,low), tolerance=1e-7)))

augmented <- kernel_call("summarize_causal", payload=list(method="augmented_treatment_contrast",
  outcomes=c(0,4), treatment=c(0,1), propensities=c(0.5,0.5),
  treated_predictions=c(2,2), control_predictions=c(0,0)))$result
stopifnot(isTRUE(all.equal(augmented$estimate, 4)),
  isTRUE(all.equal(augmented$influence_standard_error, 2)))
augmented_att <- kernel_call("summarize_causal", payload=list(method="augmented_treatment_contrast",
  estimand="att", outcomes=c(1,4), treatment=c(0,1), propensities=c(0.5,0.5),
  control_predictions=c(0,1)))$result
stopifnot(identical(augmented_att$estimand, "att"), isTRUE(all.equal(augmented_att$estimate, 2)),
  isTRUE(all.equal(augmented_att$influence_standard_error, 2)))

stratum_ate <- kernel_call("summarize_causal", payload=list(method="cross_fitted_stratum_ate",
  outcomes=c(0,4,2,6), treatment=c(0,1,0,1), strata=rep("a",4), folds=c(1L,2L,2L,1L)))$result
stopifnot(isTRUE(all.equal(unlist(stratum_ate$control_predictions), c(2,0,0,2))),
  isTRUE(all.equal(unlist(stratum_ate$treated_predictions), c(4,6,6,4))),
  isTRUE(all.equal(stratum_ate$contrast$estimate, 4)))
clustered_stratum_payload <- list(method="cross_fitted_stratum_ate", outcomes=c(0,4,2,6),
  treatment=c(0,1,0,1), strata=rep("a",4), folds=c(1L,2L,2L,1L), clusters=c("a","b","b","a"))
clustered_stratum <- kernel_call("summarize_causal", payload=clustered_stratum_payload)$result$contrast
stopifnot(clustered_stratum$cluster_count == 2, isTRUE(all.equal(clustered_stratum$influence_standard_error, 2)))
clustered_att_payload <- modifyList(clustered_stratum_payload, list(method="cross_fitted_stratum_att"))
clustered_att <- kernel_call("summarize_causal", payload=clustered_att_payload)$result$contrast
stopifnot(identical(clustered_att$estimand, "att"), isTRUE(all.equal(clustered_att$estimate, 4)),
  clustered_att$cluster_count == 2, isTRUE(all.equal(clustered_att$influence_standard_error, 2)))
clustered_stratum_payload$clusters <- c("a","a","b","b")
split_cluster_error <- tryCatch({kernel_call("summarize_causal", payload=clustered_stratum_payload); ""}, error=conditionMessage)
stopifnot(grepl("single cross-fitting fold", split_cluster_error, fixed=TRUE))

stratum_payload <- list(outcomes=rep(c(0,0,2,10,19,19,19),2),
  treatment=rep(c(0,0,1,0,1,1,1),2), strata=rep(c(rep("low",3),rep("high",4)),2),
  folds=rep(c(1L,2L), each=7))
for (estimand in c("ate", "att")) {
  stratum_payload$method <- paste0("cross_fitted_stratum_", estimand)
  contrast <- kernel_call("summarize_causal", payload=stratum_payload)$result$contrast
  stopifnot(identical(contrast$estimand, estimand),
    isTRUE(all.equal(contrast$estimate, if (estimand == "ate") 6 else 7.25)))
}

cluster_payload <- list(method="augmented_treatment_contrast", outcomes=c(0,4,0,4),
  treatment=c(0,1,0,1), propensities=rep(0.5,4),
  treated_predictions=rep(2,4), control_predictions=rep(0,4))
independent <- kernel_call("summarize_causal", payload=cluster_payload)$result
clustered <- kernel_call("summarize_causal",
  payload=c(cluster_payload, list(clusters=c("a","b","a","b"))))$result
stopifnot(clustered$estimate == independent$estimate, clustered$estimate == 4,
  clustered$cluster_count == 2, isTRUE(all.equal(clustered$influence_standard_error, 2)))

panel_event <- kernel_call("summarize_causal", payload=list(method="panel_event_study",
  periods=c(-2,-1,0), baseline_index=1L, treated=matrix(c(1,2,7,3,4,11), nrow=2, byrow=TRUE),
  control=matrix(c(0,1,3,4,5,7), nrow=2, byrow=TRUE)))$result$contrasts
stopifnot(panel_event[[1]]$estimate == 0, isTRUE(panel_event[[2]]$reference),
  isTRUE(all.equal(panel_event[[3]]$estimate, 4)), isTRUE(all.equal(panel_event[[3]]$standard_error, 1)))

staggered <- kernel_call("summarize_causal", payload=list(method="staggered_panel_att",
  periods=c(0,1,2), outcomes=matrix(c(0,5,7,10,11,16), nrow=2, byrow=TRUE),
  adoption_indices=c(1L,2L)))$result
stopifnot(staggered$event_time[[1]]$unsupported_cohorts == 1,
  is.null(staggered$event_time[[2]]$supported_cohort_att))
staggered <- staggered$contrasts
stopifnot(staggered[[1]]$estimate == 4, staggered[[1]]$n_control == 1,
  all(vapply(staggered[2:3], function(row) {
    "estimate" %in% names(row) && is.null(row$estimate) && row$n_control == 0
  }, logical(1))))
stopifnot("standard_error" %in% names(staggered[[1]]), is.null(staggered[[1]]$standard_error))
replicated_staggered <- kernel_call("summarize_causal", payload=list(method="staggered_panel_att",
  periods=c(0,1,2), outcomes=matrix(c(0,5,7,0,7,9,10,12,16,10,12,16), nrow=4, byrow=TRUE),
  adoption_indices=list(1L,1L,2L,NULL)))$result
stopifnot(isTRUE(all.equal(replicated_staggered$event_time[[1]]$supported_cohort_att, 8/3)))
stopifnot(is.null(replicated_staggered$event_time[[1]]$standard_error))
replicated_staggered <- replicated_staggered$contrasts
stopifnot(isTRUE(all.equal(replicated_staggered[[1]]$estimate, 4)),
  isTRUE(all.equal(replicated_staggered[[1]]$standard_error, 1)), replicated_staggered[[2]]$n_control == 1)
shared_staggered <- kernel_call("summarize_causal", payload=list(method="staggered_panel_att",
  periods=c(0,1,2), outcomes=matrix(c(0,5,5,0,5,5,0,1,6,0,1,6,0,0,0,0,2,4), nrow=6, byrow=TRUE),
  adoption_indices=list(1L,1L,2L,2L,NULL,NULL), include_event_time_covariance=TRUE,
  simultaneous_bands=list(draws=100L,seed=7L)))$result
shared_error <- shared_staggered$event_time[[1]]$standard_error
covariance <- shared_staggered$event_time_covariance
stopifnot(isTRUE(all.equal(unlist(covariance[[1]]), c(shared_error^2, 2*shared_error))),
  isTRUE(all.equal(unlist(covariance[[2]]), c(2*shared_error, 4))))
stopifnot(isTRUE(all.equal(shared_error, (1/sqrt(48) + 1/sqrt(8)) * sqrt(2))))
bands <- shared_staggered$simultaneous_bands
interval <- bands$intervals[[2]]$interval
stopifnot(isTRUE(all.equal(bands$critical_value, sqrt(2))),
  isTRUE(all.equal(interval$lower, 3 - 2*sqrt(2))),
  isTRUE(all.equal(interval$upper, 3 + 2*sqrt(2))))
balanced_staggered <- kernel_call("summarize_causal", payload=list(method="staggered_panel_att",
  periods=c(0,1,2), outcomes=matrix(c(0,5,7,0,7,9,10,12,16,10,12,16), nrow=4, byrow=TRUE),
  adoption_indices=list(1L,1L,2L,NULL), balanced_horizon=1L))$result
stopifnot(balanced_staggered$balanced_event_time$excluded_cohorts == 1,
  isTRUE(all.equal(vapply(balanced_staggered$event_time, function(row) as.numeric(row$supported_cohort_att), numeric(1)), c(4,2))),
  identical(balanced_staggered$contrasts, replicated_staggered))
placebo <- kernel_call("summarize_causal", payload=list(method="staggered_panel_att",
  periods=c(0,1,2), outcomes=matrix(c(0,5,7,10,11,16,0,2,4), nrow=3, byrow=TRUE),
  adoption_indices=list(1L,2L,NULL), include_pre_trends=TRUE))$result$pre_trends
stopifnot(length(placebo) == 1, placebo[[1]]$n_control == 1, placebo[[1]]$baseline_period == 1,
  isTRUE(all.equal(placebo[[1]]$estimate, 1)))
for (group in c("not_yet_treated", "never_treated")) {
  result <- kernel_call("summarize_causal", payload=list(method="staggered_panel_att",
    periods=c(0,1,2), outcomes=matrix(c(0,5,7,10,11,16,0,2,4), nrow=3, byrow=TRUE),
    adoption_indices=list(1L,2L,NULL), control_group=group))$result$contrasts[[1]]
  never <- group == "never_treated"
  stopifnot(result$n_control == if (never) 1 else 2,
    isTRUE(all.equal(result$estimate, if (never) 3 else 3.5)))
}

long_panel <- kernel_call("summarize_causal", payload=list(method="staggered_long_panel_att",
  unit_ids=c("z","a","z","a","a","z"), time=c(20,10,10,30,20,30),
  outcomes=c(11,0,10,7,5,16), adoption_times=list(a=20,z=30)))$result
stopifnot(identical(unlist(long_panel$unit_order), c("a","z")),
  isTRUE(all.equal(long_panel$contrasts[[1]]$estimate, 4)))

ranked_scenarios <- Map(function(id, capacity) list(id=id, model_key="logistic",
  simulation=list(inputs=list(time=c(1,2,4)), parameters=list(L=capacity,k=1,x0=1))),
  c("base","alt","tie"), c(100,200,200))
ranking <- kernel_call("execute_scenario_grid", payload=list(scenarios=unname(ranked_scenarios),
  rank_scenarios=c("tie","base","alt")))$result$ranking$rows
stopifnot(identical(vapply(ranking, function(row) row$id, character(1)), c("tie","alt","base")),
  all(vapply(ranking, function(row) as.numeric(row$place), numeric(1)) == c(1,1,3)))
area <- kernel_call("execute_scenario_grid", payload=list(scenarios=unname(ranked_scenarios),
  rank_scenarios=c("tie","base","alt"), rank_metric="adoption_area"))$result
path <- unlist(area$executions[[2]]$result$values)
stopifnot(identical(area$ranking$metric, "adoption_area"),
  isTRUE(all.equal(area$ranking$rows[[1]]$adoption_area, (path[1]+path[2])/2 + path[2]+path[3])))
threshold <- kernel_call("execute_scenario_grid", payload=list(scenarios=unname(ranked_scenarios),
  rank_scenarios=c("tie","base","alt"), rank_metric="time_to_threshold", rank_threshold=150))$result$ranking$rows
stopifnot(identical(vapply(threshold, function(row) row$id, character(1)), c("tie","alt","base")),
  threshold[[1]]$place == 1, threshold[[2]]$place == 1, is.null(threshold[[3]]$place),
  isTRUE(all.equal(threshold[[1]]$time_to_threshold, 2+2*(150-path[2])/(path[3]-path[2]))),
  is.null(threshold[[3]]$time_to_threshold))
ranked_scenarios[[4]] <- modifyList(ranked_scenarios[[1]], list(id="base2"))
comparison <- list(pairs=list(list(baseline="base",alternative="alt"),
  list(baseline="tie",alternative="base2")), interval_mass=0.5, evaluation_metric="adoption_area")
paired <- kernel_call("execute_scenario_grid", payload=list(scenarios=unname(ranked_scenarios),
  comparisons=list(comparison)))$result$comparisons[[1]]$result
expected <- area$ranking$rows[[1]]$adoption_area/2
stopifnot(isTRUE(all.equal(unlist(paired$evaluation_horizon), c(1,4))),
  isTRUE(all.equal(unlist(paired$differences), c(expected,-expected))), paired$mean_difference == 0)
comparison$evaluation_metric <- NULL
comparison$include_path_covariance <- TRUE
endpoint <- kernel_call("execute_scenario_grid", payload=list(scenarios=unname(ranked_scenarios),
  comparisons=list(comparison)))$result$comparisons[[1]]$result
rows <- endpoint$path_summary$rows
stopifnot(all(vapply(rows, function(row) as.numeric(row$time), numeric(1)) == c(1,2,4)))
for (field in c("mean_difference","mean_standard_error","empirical_interval")) {
  stopifnot(identical(rows[[3]][[field]], endpoint[[field]]))
}
errors <- vapply(rows, function(row) as.numeric(row$mean_standard_error), numeric(1))
for (i in seq_along(errors)) {
  stopifnot(isTRUE(all.equal(unlist(endpoint$path_covariance[[i]]), errors[i] * errors)))
}

comparison$include_path_covariance <- NULL
comparison$interval_mass <- 0.9
comparison$exact_sign_flip <- TRUE
comparison$bootstrap <- list(draws=1000, seed=42)
inference <- kernel_call("execute_scenario_grid", payload=list(scenarios=unname(ranked_scenarios),
  comparisons=list(comparison)))$result$comparisons[[1]]$result
effect <- endpoint$differences[[1]]
stopifnot(inference$sign_flip$assignments == 4, inference$sign_flip$extreme_assignments == 4,
  inference$sign_flip$p_value == 1, inference$bootstrap$seed == 42,
  isTRUE(all.equal(unlist(inference$bootstrap$mean_interval), c(-effect,effect))))

# The structured request must preserve adjacent binary64 timestamps.
for (time in list(c(1.5, 1.5 + 2^-52, 1.5 + 4 * 2^-52),
                 c(-.Machine$double.xmax, -.Machine$double.xmax / 2, .Machine$double.xmax))) {
  categories <- kernel_call("categorize_adopters",
    payload=list(time=time, adoption_rate=c(1, 2, 1)))$result
  stopifnot(identical(unlist(categories$time), time),
    identical(unlist(categories$category), c("Early Adopters", "Early Majority", "Laggards")))
}

for (origin in c(0, 1024)) {
  categories <- kernel_call("categorize_adopters",
    payload=list(time=origin + 0:3, adoption_rate=c(0, 1, 0, 1)))$result$category
  stopifnot(identical(unlist(categories),
    c("Innovators", "Early Adopters", "Early Majority", "Late Majority")))
}

for (engine in c("kairos_des", "kairos_abm", "kairos_network")) {
  payload <- jsonlite::fromJSON(fixture_path("kairos_network_checkpoint.json"), simplifyVector=FALSE)
  if (engine != "kairos_network") {
    payload$simulation_options <- list(engine=engine, seed=42, population=6, ticks_per_unit=1)
  }
  expected <- kernel_simulate_model("bass", payload)$result
  for (tick in c(0, 4, 6)) {
    payload$simulation_options$pause_at_tick <- tick
    paused <- kernel_simulate_model("bass", payload)$result
    stopifnot(identical(paused$status, "paused"))
    payload$simulation_options$pause_at_tick <- NULL
    payload$simulation_options$resume_checkpoint <- paused$checkpoint
    stopifnot(identical(kernel_simulate_model("bass", payload)$result, expected))
    payload$simulation_options$resume_checkpoint <- NULL
  }
}

simulation <- jsonlite::fromJSON(fixture_path("kairos_network_checkpoint.json"), simplifyVector=FALSE)
paused <- simulation
paused$simulation_options$pause_at_tick <- 6
scenarios <- list(list(id="complete", model_key="bass", simulation=simulation),
                  list(id="paused", model_key="bass", simulation=paused))
response_json <- kernel_call("execute_scenario_grid", payload=list(scenarios=scenarios), raw=TRUE)
response <- jsonlite::fromJSON(response_json, simplifyVector=FALSE)
stopifnot(is.character(response_json), identical(response$result$executions[[2]]$result$status, "paused"),
  rawToChar(kernel_scenario_executions_ipc(response_json)[1:6]) == "ARROW1")

lower <- 1.5
upper <- lower + .Machine$double.eps
balance_payload <- list(method="matched_treatment_contrast", outcomes=c(0,0,1,1,1),
  treatment=c(0,0,1,1,1), scores=c(0,1,0,0,1), caliper=0,
  balance_covariates=list(narrow=c(lower,upper,lower,upper,lower), constant=rep(1,5)))
balance <- kernel_call("summarize_causal", payload=balance_payload)$result$covariate_balance
stopifnot(isTRUE(all.equal(balance$narrow$before_matching$standardized_mean_difference, (-1/6)/sqrt(7/18))),
  balance$narrow$standardized_mean_difference == 0,
  isTRUE(all.equal(balance$narrow$before_matching$mean_difference / (upper - lower), -1/6)),
  is.null(balance$constant$standardized_mean_difference))

observations <- list(outcomes=c(0,4,2,6), treatment=c(0,1,0,1))
fit <- kernel_call("summarize_causal", payload=c(observations, list(
  method="cross_fitted_stratum_ate", strata=rep("a",4), folds=c(1,2,2,1))))$result
matched <- kernel_call("summarize_causal", payload=c(observations, list(
  method="matched_treatment_contrast", scores=c(0,0,1,0), caliper=0,
  control_predictions=fit$control_predictions)))$result
stopifnot(matched$estimate == 5, matched$adjusted_estimate == 6,
  is.null(matched$conditional_standard_error))

correlated_payload <- list(method="matched_treatment_contrast", outcomes=c(0,10,12,3,99),
  treatment=c(0,0,1,1,0), scores=c(0,2,2,2,0), caliper=1,
  outcome_variances=c(1,4,1,1,1), outcome_covariance_factors=list(rep(7,5),c(0,0,2,2,0)))
correlated <- kernel_call("summarize_causal", payload=correlated_payload)$result
stopifnot(isTRUE(all.equal(correlated$conditional_standard_error, sqrt(8.5))),
  correlated$uncertainty_model$model == "diagonal_plus_factors",
  correlated$uncertainty_model$factor_count == 2)

fitted_covariance <- kernel_call("summarize_causal", payload=list(
  method="matched_treatment_contrast", outcomes=c(0,4,2,6),
  treatment=c(0,1,0,1), scores=c(0,0,1,0), caliper=0,
  outcome_variances=rep(1,4),
  outcome_covariance_factors=list(rep(7,4), c(0,2,0,2)),
  fit_controls=list(strata=rep("a",4), folds=c(1,2,2,1))))$result
stopifnot(fitted_covariance$estimate == 5, fitted_covariance$adjusted_estimate == 6,
  identical(as.numeric(unlist(fitted_covariance$adjusted_outcome_weights)),
    c(0,-1.5,1,0.5,2,0.5,3,0.5)),
  isTRUE(all.equal(fitted_covariance$conditional_standard_error, sqrt(5.5))),
  isTRUE(all.equal(fitted_covariance$adjusted_standard_error, sqrt(7))))

fitted_scores <- kernel_call("summarize_causal", payload=list(
  method="matched_treatment_contrast", outcomes=c(0,4,2,6),
  treatment=c(0,1,0,1), caliper=0,
  fit_scores=list(covariates=rep(list(list(0)),4), folds=c(1,1,2,2), ridge=1)))$result
stopifnot(fitted_scores$estimate == 5,
  identical(unlist(fitted_scores$score_fit$propensities), rep(0.5,4)))
stopifnot(fitted_scores$score_uncertainty_scope == "conditional_on_fitted_scores_and_pairs",
  fitted_scores$match_diagnostics$minimum_matched_caliper_slack == 0)
stopifnot(fitted_scores$match_diagnostics$minimum_runner_up_distance_gap == 0)

score_mixture <- kernel_call("summarize_causal", payload=list(
  method="matched_treatment_contrast", outcomes=c(0,4,2), treatment=c(0,1,0),
  scores=c(0,0,1), caliper=0, score_draws=list(c(0,0,1), c(1,0,0)),
  outcome_variances=c(0,1,6), control_predictions=c(0,0,2)))$result$score_sensitivity
stopifnot(all(vapply(score_mixture$draws, function(draw) draw$estimate, numeric(1)) == c(4,2)),
  isTRUE(score_mixture$same_treated_subset),
  isTRUE(all.equal(score_mixture$mixture_standard_deviation, sqrt(5))),
  isTRUE(all.equal(score_mixture$adjusted_mixture_standard_deviation, 2)))

score_refits <- kernel_call("summarize_causal", payload=list(
  method="matched_treatment_contrast", outcomes=c(0,4,2,6,8,12,10,14),
  treatment=rep(c(0,1),4), caliper=1, outcome_variances=rep(1,8),
  fit_scores=list(covariates=rep(list(list(0)),8), folds=rep(c(1,2), each=4), ridge=1),
  score_refit_omit_indices=c(0,4)))$result$score_sensitivity
stopifnot(all(unlist(score_refits$omitted_training_indices) == c(0,4)),
  all(vapply(score_refits$draws[[1]]$pairs, function(pair) pair$control_index, numeric(1)) == c(0,0,4,4)),
  "mixture_standard_deviation" %in% names(score_refits),
  is.null(score_refits$mixture_standard_deviation))

insufficient_draw <- kernel_call("summarize_causal", payload=list(
  method="matched_treatment_contrast", outcomes=c(0,4), treatment=c(0,1),
  scores=c(0,0), caliper=0, score_draws=list(c(0,0),c(0,1)),
  outcome_variances=c(1,1)))$result$score_sensitivity
stopifnot(identical(vapply(insufficient_draw$draws, function(draw) draw$status, character(1)),
    c("matched","insufficient_matches")),
  insufficient_draw$draws[[1]]$estimate == 4,
  "mixture_standard_deviation" %in% names(insufficient_draw),
  is.null(insufficient_draw$mixture_standard_deviation))

bootstrap_payload <- list(method="matched_treatment_contrast", outcomes=c(0,4,2,6,8,12,10,14),
  treatment=rep(c(0,1),4), caliper=1, outcome_variances=rep(1,8),
  fit_scores=list(covariates=lapply(c(0,3,2,5,1,4,3,6), list),
    folds=rep(c(1,2), each=4), ridge=1),
  score_bootstrap=list(draws=8, seed=17))
run_score_bootstrap <- function() {
  kernel_call("summarize_causal", payload=bootstrap_payload)$result$score_sensitivity
}
score_bootstrap <- run_score_bootstrap()
stopifnot(identical(score_bootstrap, run_score_bootstrap()),
  score_bootstrap$seed == 17, length(score_bootstrap$draws) == 8,
  score_bootstrap$mixture_standard_deviation > 0)

local({
  selection <- list(name="provider-fixture", expected_version="1", extension_point="dataset_provider")
  manifest <- list(name=selection$name, version="1", entrypoint="fixture:provide",
    stability="stable", extension_points=list("dataset_provider"))
  kernel_call("register_extension_manifest", payload=list(manifest=manifest))
  tryCatch({
    dataset <- list(kind="adoption", time=c(0,1), adoption=c(1,2),
      provenance=list(source="fixture", license="CC-BY-4.0", extraction_time="2026-11-07T00:00:00Z"))
    artifact <- invoke_registered_extension(function(resolved, data) data, selection, dataset)
    stopifnot(isTRUE(artifact$validation$ok), artifact$dataset$provenance$checksum ==
      "fada5d71f8981fa746be73b26de45ad470fa4e74f6a7d7a48234f4f50de0a76a")
    edges <- list(kind="network_edges", source="001", target="1", weight=0.5,
      provenance=dataset$provenance)
    network <- invoke_registered_extension(function(resolved, data) data, selection, edges)
    stopifnot(isTRUE(network$validation$ok),
      identical(network$dataset$source, list("001")),
      identical(network$dataset$target, list("1")),
      identical(network$dataset$weight, list(0.5)),
      identical(network$dataset$provenance$source, "fixture"))
    failure <- tryCatch(invoke_registered_extension(function(...) list(kind="adoption"),
      selection, NULL), error=conditionMessage)
    stopifnot(grepl("requires dataset provenance", failure, fixed=TRUE))
  }, finally=kernel_call("unregister_extension_manifest",
    payload=list(name=selection$name, expected_version="1")))
})
