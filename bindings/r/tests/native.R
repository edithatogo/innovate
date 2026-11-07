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
stopifnot(all(unlist(net$counts)==c(0,0,3)),all(vapply(net$completions,function(e)e$tick,numeric(1))==c(6,11,16)))
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
weighted_network$simulation_options$schedule <- list(list(tick=6,edges=list(c(1,2)),service_ticks=2))
weighted_network$simulation_options$agent_priorities <- c(-2,0,3)
weighted_result <- kernel_simulate_model("bass",weighted_network)$result
stopifnot(all(unlist(weighted_result$counts)==c(1,1,3)),all(vapply(weighted_result$completions,function(e)e$tick,numeric(1))==c(6,9)))
stopifnot(all(unlist(weighted_result$metadata$agent_priorities)==c(-2,0,3)))
cat("R directed weighted Kairos resource and priorities passed\n")
