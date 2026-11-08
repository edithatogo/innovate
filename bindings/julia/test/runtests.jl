using Test, Innovate
@testset "Native conformance" begin
    payload = Dict("inputs" => Dict("time" => [0.,1.,2.]), "parameters" => Dict("L"=>100.,"k"=>1.,"x0"=>1.))
    result = kernel_predict_model("logistic", payload)
    @test result["metadata"]["runtime"] == "rust_native"
    @test result["result"]["values"][2] == 50.0
    @test length(kernel_discover_models()["models"]) == 16
    @test_throws ErrorException kernel_predict_model("not_a_model", payload)
    for _ in 1:100
        @test kernel_predict_model("logistic",payload)["result"]["values"][2] == 50.0
    end
end

@testset "Five model analytic oracles" begin
    bass(t) = 100*(1-exp(-0.38*t))/(1+0.35/0.03*exp(-0.38*t))
    for (name, parameters, expected) in [
        ("logistic",Dict("L"=>100.,"k"=>0.6,"x0"=>8.),100/(1+exp(-0.6*(3-8)))),
        ("gompertz",Dict("a"=>100.,"b"=>3.,"c"=>0.3),100*exp(-3*exp(-0.3*3))),
        ("fisher_pry",Dict("alpha"=>0.6,"t0"=>8.),1/(1+exp(-0.6*(3-8)))),
        ("bass",Dict("p"=>0.03,"q"=>0.35,"m"=>100.),bass(3)),
        ("norton_bass",Dict("p1"=>0.03,"q1"=>0.35,"m1"=>100.),bass(3))]
        response = kernel_predict_model(name, Dict("inputs"=>Dict("time"=>[3.,4.]),"parameters"=>parameters))
        @test response["schema_version"] == "2.0"
        result = response["result"]
        values = haskey(result,"values") ? result["values"] : [r[1] for r in result["rows"]]
        @test isapprox(values[1], expected; atol=1e-10,rtol=0)
    end
end

@testset "Multi-generation fit and forecast" begin
    fraction(t,p,q) = t<=0 ? 0. : (1-exp(-(p+q)*t))/(1+q/p*exp(-(p+q)*t))
    oracle(t) = begin
        f1=fraction(t,0.03,0.2); f2=fraction(t-4,0.02,0.3)
        [1000*f1*(1-f2),(1500+1000*f1)*f2]
    end
    time=collect(0.:0.5:20.)
    options=Dict("n_generations"=>2,"variant"=>"successive_generations","launch_times"=>[0.,4.])
    fit=kernel_fit_model("norton_bass", Dict("inputs"=>Dict("time"=>time,"observed"=>[oracle(t) for t in time]),"constructor_kwargs"=>options))
    @test fit["result"]["diagnostics"]["optimizer"]["converged"]
    @test fit["result"]["diagnostics"]["identifiability"]["rank"] == 6
    result=kernel_predict_model("norton_bass",Dict("inputs"=>Dict("time"=>[21.,25.,30.]),"state"=>fit["result"]["state"]))
    for (i,t) in enumerate([21.,25.,30.])
        @test isapprox(result["result"]["rows"][i],oracle(t);atol=0.01,rtol=0)
    end
    @test result["result"]["rates"][3][1] < 0
    @test kernel_predict_model("logistic",Dict("inputs"=>Dict("time"=>[1.]),"parameters"=>Dict("L"=>100.,"k"=>1.,"x0"=>1.)))["result"]["values"][1] == 50
end

@testset "Competitive analytic limits" begin
    bass(t)=100*(1-exp(-0.38*t))/(1+0.35/0.03*exp(-0.38*t))
    for (name,parameters,options,expected) in [
        ("lotka_volterra",Dict("alpha1"=>0.3,"beta1"=>0.,"alpha2"=>0.4,"beta2"=>0.),Dict("initial_values"=>[0.05,0.02]),[1/(1+19*exp(-0.3*3)),1/(1+49*exp(-0.4*3))]),
        ("multi_product",Dict("p1"=>0.03,"q1"=>0.35,"m1"=>100.,"p2"=>0.03,"q2"=>0.35,"m2"=>200.,"alpha_1_2"=>0.,"alpha_2_1"=>0.),Dict("n_products"=>2),[bass(3),2*bass(3)])]
        result=kernel_predict_model(name,Dict("inputs"=>Dict("time"=>[3.]),"parameters"=>parameters,"constructor_kwargs"=>options))["result"]
        @test length(result["rows"][1])==2
        @test isapprox(result["rows"][1],expected;atol=1e-6,rtol=0)
        @test length(result["rates"][1])==2
    end
end

@testset "Native numerical overflow guard" begin
    @test_throws r"finite numerical domain" kernel_fit_model("logistic",Dict("inputs"=>Dict("time"=>[0.,1.,2.,3.],"observed"=>fill(1e-12,4)),"fit_options"=>Dict("initial_parameters"=>Dict("L"=>1e140,"k"=>1e15,"x0"=>1.))))
end
@testset "Kairos native replay" begin
    payload=Dict("inputs"=>Dict("time"=>[0.,1.,5.,20.]),"parameters"=>Dict("p"=>0.2,"q"=>0.3,"m"=>100.),"simulation_options"=>Dict("engine"=>"kairos_des","seed"=>42,"population"=>100))
    a=kernel_simulate_model("bass",payload)
    @test a==kernel_simulate_model("bass",payload)
    @test a["result"]["counts"][1]==0
    @test a["metadata"]["engine"]=="kairos_des"
end
@testset "ABM forcing and agent conservation" begin
    payload=Dict("inputs"=>Dict("time"=>[0.,1.,3.,5.]),"parameters"=>Dict("p"=>0.2,"q"=>0.3,"m"=>100.),"simulation_options"=>Dict("engine"=>"kairos_abm","seed"=>42,"population"=>100,"covariate"=>Dict("time"=>[0.,3.],"values"=>[-1.,0.],"beta_p"=>0.2,"beta_q"=>0.3)))
    a=kernel_simulate_model("bass",payload)["result"]
    @test a["counts"][1:3]==[0,0,0]
    @test a["counts"][4]>0
    @test length(a["agents"])==100
    @test count(x->x["adopted"],a["agents"])==a["counts"][4]
end
@testset "Historical Logistic and initial ABM" begin
    p=Dict("inputs"=>Dict("time"=>[1.],"covariates"=>Dict("price"=>[1.])),"parameters"=>Dict("L"=>100.,"k"=>0.5,"x0"=>2.,"beta_L_price"=>10.,"beta_k_price"=>0.1,"beta_x0_price"=>0.2),"constructor_kwargs"=>Dict("covariates"=>["price"]))
    @test isapprox(kernel_predict_model("logistic",p)["result"]["values"][1],110/(1+exp(0.72));atol=1e-10)
    p=Dict("inputs"=>Dict("time"=>[0.,20.]),"parameters"=>Dict("p"=>0.2,"q"=>0.3,"m"=>100.),"simulation_options"=>Dict("engine"=>"kairos_abm","seed"=>42,"population"=>100,"initial_adopters"=>20))
    @test kernel_simulate_model("bass",p)["result"]["counts"][1]==20
end
@testset "Complementary goods and network FIFO" begin
    p=Dict("inputs"=>Dict("time"=>[0.,5.,20.]),"parameters"=>Dict("p"=>1e12,"q"=>0.,"m"=>3.),"simulation_options"=>Dict("engine"=>"kairos_network","population"=>3,"seed"=>42,"edges"=>[],"capacity"=>1,"service_ticks"=>5,"ticks_per_unit"=>1))
    a=kernel_simulate_model("bass",p)["result"]
    @test a["counts"]==[0,0,3]
    @test [e["tick"] for e in a["completions"]]==[6,11,16]
    @test a["resource_metrics"]["peak_queue_length"] == 2
    @test a["resource_metrics"]["peak_in_service"] == 1
    @test a["resource_metrics"]["idle_slot_ticks"] == 5
    @test a["resource_metrics"]["over_capacity_slot_ticks"] == 0
    closure = deepcopy(p)
    closure["simulation_options"]["schedule"] = [Dict("tick"=>3,"capacity"=>0)]
    closed = kernel_simulate_model("bass",closure)["result"]
    @test closed["resource_metrics"]["idle_slot_ticks"] == 1
    @test closed["resource_metrics"]["over_capacity_slot_ticks"] == 3
    @test only(closed["completions"])["tick"] == 6
    p["simulation_options"]["max_wait_ticks"]=4
    expired=kernel_simulate_model("bass",p)["result"]
    @test expired["counts"]==[0,0,2]
    @test only(expired["abandonments"])["tick"]==7
    p=Dict("inputs"=>Dict("time"=>[0.,3.]),"parameters"=>Dict("k1"=>0.3,"k2"=>0.4,"c1"=>0.,"c2"=>0.),"constructor_kwargs"=>Dict("initial_values"=>[0.05,0.02]))
    @test isapprox(kernel_predict_model("complementary_goods",p)["result"]["rows"][2][1],1/(1+19*exp(-0.9));atol=1e-8)
end
@testset "Complementary analytic fit" begin
    capacity=0.3/0.27
    time=collect(0.:25.)
    y=[fill(capacity/(1+(capacity/0.05-1)*exp(-0.3*t)),2) for t in time]
    p=Dict("inputs"=>Dict("time"=>time,"observed"=>y),"fit_options"=>Dict("initial_parameters"=>Dict("k1"=>0.28,"k2"=>0.28,"c1"=>0.025,"c2"=>0.025)))
    f=kernel_fit_model("complementary_goods",p)["result"]
    @test f["diagnostics"]["optimizer"]["converged"]
    @test abs(f["parameters"]["c1"]-0.03)<1e-4
end
@testset "Lock-in analytic forecast" begin
    p=Dict("inputs"=>Dict("time"=>[10.,20.]),"parameters"=>Dict("alpha1"=>0.3,"alpha2"=>0.2,"beta1"=>0.,"beta2"=>0.,"gamma1"=>0.,"gamma2"=>0.,"m"=>1000.),"constructor_kwargs"=>Dict("initial_values"=>[10.,0.]))
    a=kernel_predict_model("lock_in",p)["result"]
    @test isapprox(a["rows"][1][1],1000/(1+99*exp(-3));atol=1e-5)
    @test a["rows"][1][2]==0
end
@testset "Historical weighted network" begin
    p=Dict("inputs"=>Dict("time"=>[0.,3.]),"constructor_kwargs"=>Dict("base_model"=>"logistic","adjacency"=>[[0.,2.],[0.,0.]],"node_labels"=>["a","b"]),"parameters"=>Dict("spillover_strength"=>0.2,"node_0_L"=>100.,"node_0_k"=>1.,"node_0_x0"=>3.,"node_1_L"=>200.,"node_1_k"=>1.,"node_1_x0"=>3.))
    a=kernel_predict_model("network_diffusion",p)["result"]
    @test isapprox(a["rows"][2][1],70.;atol=1e-10)
    @test a["rows"][2][2]==100
end
@testset "Policy event boundary" begin
    p=Dict("inputs"=>Dict("time"=>[0.,2.,3.]),"parameters"=>Dict("decay"=>0.5,"base_L"=>100.,"base_k"=>1.,"base_x0"=>2.),"constructor_kwargs"=>Dict("base_model"=>"logistic","event_times"=>[2.],"event_effects"=>[0.5],"event_labels"=>["policy"]))
    @test kernel_predict_model("policy_hazard",p)["result"]["values"][2]==75
end
@testset "Heterogeneous network service" begin
    p=Dict("inputs"=>Dict("time"=>[0.,100.]),"parameters"=>Dict("p"=>1e12,"q"=>0.,"m"=>3.),"simulation_options"=>Dict("engine"=>"kairos_network","population"=>3,"seed"=>42,"edges"=>[],"capacity"=>1,"service_ticks"=>5,"ticks_per_unit"=>1,"innovation_multipliers"=>[1.,0.,1.],"agent_service_ticks"=>[4,9,7]))
    a=kernel_simulate_model("bass",p)["result"]
    @test a["counts"]==[0,2]
    expected=1
    for (request,completion) in zip(a["requests"],a["completions"])
        @test request["agent"]==completion["agent"]
        expected += [4,9,7][request["agent"]+1]
        @test completion["tick"]==expected
    end
end
@testset "Latent saved residual state" begin
    p=Dict("inputs"=>Dict("time"=>[0.,2.,4.]),"parameters"=>Dict("base_L"=>100.,"base_k"=>1.,"base_x0"=>2.,"latent_smoothing"=>0.3,"latent_noise"=>0.),"constructor_kwargs"=>Dict("base_model"=>"logistic","latent_time"=>[0.,4.],"latent_values"=>[2.,6.]))
    a=kernel_predict_model("latent_process",p)["result"]
    @test a["values"][2]==54.
    time=collect(0.:20.)
    y=[100/(1+exp(-0.5*(t-8)))+2+0.1*sin(t) for t in time]
    f=kernel_fit_model("latent_process",Dict("inputs"=>Dict("time"=>time,"observed"=>y),"constructor_kwargs"=>Dict("base_model"=>"logistic","smoothing"=>0.4)))["result"]
    @test length(f["state"]["constructor_kwargs"]["latent_values"])==21
    q=f["parameters"]
    expected=q["base_L"]/(1+exp(-q["base_k"]*(30-q["base_x0"])))+f["state"]["constructor_kwargs"]["latent_values"][end]
    a=kernel_predict_model("latent_process",Dict("inputs"=>Dict("time"=>[30.]),"state"=>f["state"]))["result"]
    @test isapprox(a["values"][1],expected;atol=1e-10)
end
@testset "Fixed time regime forecast" begin
    time=collect(0.:29.)
    y=[t<15 ? 100/(1+exp(-0.4*(t-6))) : 200/(1+exp(-0.3*(t-18))) for t in time]
    f=kernel_fit_model("regime_switching",Dict("inputs"=>Dict("time"=>time,"observed"=>y),"constructor_kwargs"=>Dict("base_model"=>"logistic","prediction_mode"=>"fixed_time","split_index"=>15)))["result"]
    @test f["state"]["constructor_kwargs"]["split_time"]==15
    @test isapprox(f["parameters"]["regime_1_k"],0.3;atol=0.001)
    a=kernel_predict_model("regime_switching",Dict("inputs"=>Dict("time"=>[20.,30.,40.]),"state"=>f["state"]))["result"]
    b=kernel_predict_model("regime_switching",Dict("inputs"=>Dict("time"=>[40.]),"state"=>f["state"]))["result"]
    @test a["values"][end]==b["values"][1]
end
@testset "Short Fisher-Pry regime" begin
    f=kernel_fit_model("regime_switching",Dict("inputs"=>Dict("time"=>[0.,1.],"observed"=>[0.11920292202211755,0.2689414213699951]),"constructor_kwargs"=>Dict("base_model"=>"fisher_pry")))["result"]
    @test f["details"]["regime_count"]==1
end
@testset "Centered hierarchical group state" begin
    time=collect(0.:24.)
    y=[[100/(1+exp(-0.4*(t-7))),200/(1+exp(-0.3*(t-9)))] for t in time]
    f=kernel_fit_model("hierarchical",Dict("inputs"=>Dict("time"=>time,"observed"=>y),"constructor_kwargs"=>Dict("base_model"=>"logistic","groups"=>["a","b"],"output"=>"groups")))["result"]
    @test isapprox(f["parameters"]["global_L"],150.;atol=0.01)
    a=kernel_predict_model("hierarchical",Dict("inputs"=>Dict("time"=>[30.]),"state"=>f["state"]))["result"]
    @test isapprox(a["rows"][1][1],100/(1+exp(-9.2));atol=0.01)
    @test isapprox(a["rows"][1][2],200/(1+exp(-6.3));atol=0.01)
end
@testset "Weighted scalar and Gaussian mixture" begin
    time=collect(0.:24.)
    y=[100/(1+exp(-0.4*(t-7))) for t in time];y[21]=300
    weights=ones(25);weights[21]=0
    f=kernel_fit_model("logistic",Dict("inputs"=>Dict("time"=>time,"observed"=>y),"fit_options"=>Dict("weights"=>weights,"initial_parameters"=>Dict("L"=>110.,"k"=>0.35,"x0"=>8.))))["result"]
    @test isapprox(f["parameters"]["L"],100.;atol=1e-5)
    time=collect(0.:0.5:24.5)
    y=[iseven(i-1) ? 100/(1+exp(-0.4*(t-6))) : 200/(1+exp(-0.3*(t-10))) for (i,t) in enumerate(time)]
    parameters=Dict("model_0_L"=>100.,"model_0_k"=>0.4,"model_0_x0"=>6.,"model_1_L"=>200.,"model_1_k"=>0.3,"model_1_x0"=>10.,"weight_0"=>0.4,"weight_1"=>0.6,"noise_scale"=>2.)
    f=kernel_fit_model("mixture",Dict("inputs"=>Dict("time"=>time,"observed"=>y),"constructor_kwargs"=>Dict("components"=>["logistic","logistic"],"max_iter"=>30,"noise_floor"=>0.001),"fit_options"=>Dict("initial_parameters"=>parameters)))["result"]
    @test f["optimizer"]["converged"]
    @test isapprox(f["parameters"]["model_1_L"],200.;atol=0.05)
    @test kernel_predict_model("mixture",Dict("inputs"=>Dict("time"=>[40.]),"state"=>f["state"]))["result"]["values"][1]>140
end
@testset "Composite native levels rates and origin" begin
    p=Dict("inputs"=>Dict("time"=>[0.,3.,10.,20.]),"constructor_kwargs"=>Dict("components"=>["bass","logistic"],"initial_values"=>[0.,10.]),"parameters"=>Dict("p_1"=>0.03,"q_1"=>0.35,"m_1"=>100.,"L_2"=>200.,"k_2"=>0.4,"x0_2"=>99.,"alpha_1_2"=>0.,"alpha_2_1"=>0.))
    a=kernel_predict_model("composite",p)["result"]
    logistic=200/(1+19*exp(-4))
    @test isapprox(a["rows"][3][2],logistic;atol=1e-6)
    @test isapprox(a["rates"][3][2],0.4*logistic*(1-logistic/200);atol=1e-6)
    time=collect(0.:24.)
    y=[100*(1-exp(-0.38*t))/(1+0.35/0.03*exp(-0.38*t)) for t in time]
    f=kernel_fit_model("composite",Dict("inputs"=>Dict("time"=>time,"observed"=>y),"constructor_kwargs"=>Dict("components"=>["bass"]),"fit_options"=>Dict("initial_parameters"=>Dict("p_1"=>0.025,"q_1"=>0.32,"m_1"=>110.))))["result"]
    @test f["diagnostics"]["optimizer"]["converged"]
    @test kernel_predict_model("composite",Dict("inputs"=>Dict("time"=>[30.]),"state"=>f["state"]))["result"]["rows"][1][1]>99.5
end
@testset "Mechanistic ODE covariates and events" begin
    p=Dict("inputs"=>Dict("time"=>[0.,1.,2.,4.],"covariates"=>Dict("dose"=>[0.,1.,2.,4.])),"constructor_kwargs"=>Dict("covariates"=>["dose"]),"parameters"=>Dict("p"=>0.1,"q"=>0.,"m"=>100.,"beta_p_dose"=>0.02,"beta_q_dose"=>0.,"beta_m_dose"=>0.))
    a=kernel_predict_model("bass",p)["result"]
    @test isapprox(a["values"][4],100*(1-exp(-0.56));atol=1e-7)
    a=kernel_predict_model("bass",Dict("inputs"=>Dict("time"=>[5.,6.],"covariates"=>Dict("dose"=>[5.,6.])),"state"=>a["state"]))["result"]
    @test isapprox(a["values"][2],100*(1-exp(-0.96));atol=1e-7)
    time=collect(0.:0.5:8.)
    boundary=100*exp(log(0.01)*exp(-0.6))
    y=[t<2 ? 100*exp(log(0.01)*exp(-0.3*t)) : 200*exp(log(boundary/200)*exp(-0.2*(t-2))) for t in time]
    f=kernel_fit_model("gompertz",Dict("inputs"=>Dict("time"=>time,"observed"=>y),"constructor_kwargs"=>Dict("t_event"=>2.,"initial_value"=>1.),"fit_options"=>Dict("initial_parameters"=>Dict("a"=>100.,"b"=>77.,"c"=>0.3,"a_post"=>200.,"b_post"=>99.,"c_post"=>0.18))))["result"]
    @test f["diagnostics"]["optimizer"]["converged"]
    @test isapprox(kernel_predict_model("gompertz",Dict("inputs"=>Dict("time"=>[10.]),"state"=>f["state"]))["result"]["values"][1],200*exp(log(boundary/200)*exp(-1.6));atol=0.001)
end
@testset "Floating forcing replay" begin
    time=collect(0.:0.5:5.);dose=[1+sin(0.7*t) for t in time]
    p=Dict("inputs"=>Dict("time"=>time,"covariates"=>Dict("dose"=>dose)),"constructor_kwargs"=>Dict("covariates"=>["dose"]),"parameters"=>Dict("p"=>0.1,"q"=>0.,"m"=>100.,"beta_p_dose"=>0.02,"beta_q_dose"=>0.,"beta_m_dose"=>0.))
    a=kernel_predict_model("bass",p)["result"]
    b=kernel_predict_model("bass",Dict("inputs"=>p["inputs"],"state"=>a["state"]))["result"]
    @test all(isapprox.(a["values"],b["values"];atol=1e-7))
end

@testset "Directed weighted Kairos resource" begin
    p=Dict("inputs"=>Dict("time"=>[0.,5.,20.]),"parameters"=>Dict("p"=>0.,"q"=>1e12,"m"=>3.),"simulation_options"=>Dict("engine"=>"kairos_network","population"=>3,"seed"=>42,"directed"=>true,"edges"=>[[0,1]],"edge_weights"=>[1e308],"initial_agents"=>[0],"capacity"=>1,"service_ticks"=>5,"ticks_per_unit"=>1))
    p["simulation_options"]["schedule"] = [Dict("tick"=>6,"edges"=>[[1,2]],"service_ticks"=>2,"capacity"=>2,"p"=>1e12,"q"=>0.)]
    p["simulation_options"]["agent_priorities"] = [-2,0,3]
    a=kernel_simulate_model("bass",p)["result"]
    @test a["metadata"]["agent_priorities"]==[-2,0,3]
    @test a["metadata"]["current_capacity"]==2
    @test a["metadata"]["current_p"]==1e12 && a["metadata"]["current_q"]==0
    @test a["counts"]==[1,1,3]
    @test [entry["tick"] for entry in a["completions"]]==[6,9]
end

@testset "Kairos queue limit and retry" begin
    options=Dict("engine"=>"kairos_network","population"=>2,"seed"=>42,"edges"=>[],"capacity"=>1,"service_ticks"=>5,"ticks_per_unit"=>1,"max_queue_length"=>0,"retry"=>Dict("delay_ticks"=>4,"max_retries"=>1))
    payload=Dict("inputs"=>Dict("time"=>[0.,6.,12.]),"parameters"=>Dict("p"=>1e12,"q"=>0.,"m"=>2.),"simulation_options"=>options)
    result=kernel_simulate_model("bass",payload)["result"]
    @test result["counts"]==[0,1,2]
    @test only(result["balks"])["tick"]==2
    @test only(result["retry_eligibility"])["tick"]==6
end

@testset "Norton Bass forcing history" begin
    parameters=Dict("p1"=>0.1,"q1"=>0.,"m1"=>100.,"p2"=>0.1,"q2"=>0.,"m2"=>100.,"beta_p1_price"=>0.05,"beta_q1_price"=>0.,"beta_m1_price"=>0.,"beta_p2_price"=>0.05,"beta_q2_price"=>0.,"beta_m2_price"=>0.)
    payload=Dict("inputs"=>Dict("time"=>[0.,1.,2.],"covariates"=>Dict("price"=>[0.,1.,2.])),"parameters"=>parameters,"constructor_kwargs"=>Dict("n_generations"=>2,"covariates"=>["price"]))
    result=kernel_predict_model("norton_bass",payload)["result"]
    @test isapprox(result["rows"][3][1],30*exp(-0.3);atol=1e-5)
    forecast=kernel_predict_model("norton_bass",Dict("inputs"=>Dict("time"=>[3.]),"state"=>result["state"]))["result"]
    @test isapprox(forecast["rows"][1][1],50*exp(-0.5);atol=1e-5)
end

@testset "Interacting Lotka forcing fit and history" begin
    fixture=Innovate.JSON.parsefile(joinpath(@__DIR__,"../../../rust/crates/innovate-core/tests/fixtures/lotka_forcing_scipy.json");dicttype=Dict{String,Any})
    start=copy(fixture["parameters"])
    start["alpha1"]=0.29
    start["beta_beta1_price"]=-0.025
    for options in (Dict("initial_parameters"=>start,"max_iterations"=>150), Dict("max_iterations"=>150))
        payload=Dict("inputs"=>Dict("time"=>fixture["time"],"observed"=>fixture["rows"],"covariates"=>fixture["covariates"]),"constructor_kwargs"=>Dict("initial_values"=>fixture["initial"],"covariates"=>["price"]),"fit_options"=>options)
        fit=kernel_fit_model("lotka_volterra",payload)["result"]
        @test fit["diagnostics"]["optimizer"]["converged"]
        forecast=kernel_predict_model("lotka_volterra",Dict("inputs"=>Dict("time"=>fixture["forecast_time"]),"state"=>fit["state"]))["result"]
        @test maximum(abs.(forecast["rows"][1].-fixture["forecast_rows"][1]))<1e-5
    end
end

@testset "Multi product forcing layouts and conditional fit" begin
    fixture = Innovate.JSON.parsefile(
        joinpath(@__DIR__, "../../../rust/crates/innovate-core/tests/fixtures/multi_forcing_scipy.json");
        dicttype=Dict{String,Any},
    )
    for layout in ["corrected", "historical"]
        payload = Dict(
            "inputs"=>Dict("time"=>fixture["time"], "covariates"=>fixture["covariates"]),
            "parameters"=>fixture["parameters"],
            "constructor_kwargs"=>Dict("n_products"=>2, "covariates"=>["price"], "coefficient_layout"=>layout),
        )
        result = kernel_predict_model("multi_product", payload)["result"]
        @test maximum(abs.(result["rows"][end].-fixture["cases"][layout]["rows"][end])) < 1e-5
        forecast = kernel_predict_model(
            "multi_product", Dict("inputs"=>Dict("time"=>fixture["forecast_time"]), "state"=>result["state"]),
        )["result"]
        @test maximum(abs.(forecast["rows"][1].-fixture["cases"][layout]["forecast_rows"][1])) < 1e-5
    end
    start = copy(fixture["parameters"])
    start["p1"] = 0.0305
    fit = kernel_fit_model("multi_product", Dict(
        "inputs"=>Dict("time"=>fixture["time"], "covariates"=>fixture["covariates"], "observed"=>fixture["cases"]["corrected"]["rows"]),
        "constructor_kwargs"=>Dict("n_products"=>2, "covariates"=>["price"]),
        "fit_options"=>Dict("initial_parameters"=>start, "max_iterations"=>200),
    ))["result"]
    @test fit["diagnostics"]["optimizer"]["converged"]
    forecast = kernel_predict_model(
        "multi_product", Dict("inputs"=>Dict("time"=>fixture["forecast_time"]), "state"=>fit["state"]),
    )["result"]
    @test maximum(abs.(forecast["rows"][1].-fixture["cases"]["corrected"]["forecast_rows"][1])) < 1e-3
end

@testset "Complementary goods forcing fit and history" begin
    fixture = Innovate.JSON.parsefile(
        joinpath(@__DIR__, "../../../rust/crates/innovate-core/tests/fixtures/complementary_forcing_scipy.json");
        dicttype=Dict{String,Any},
    )
    start = copy(fixture["parameters"])
    start["k1"] = 0.29
    start["beta_c2_price"] = -0.007
    for options in (Dict("initial_parameters"=>start, "max_iterations"=>150), Dict("max_iterations"=>150))
        fit = kernel_fit_model("complementary_goods", Dict(
            "inputs"=>Dict("time"=>fixture["time"], "covariates"=>fixture["covariates"], "observed"=>fixture["rows"]),
            "constructor_kwargs"=>Dict("initial_values"=>fixture["initial"], "covariates"=>["price"]),
            "fit_options"=>options,
        ))["result"]
        @test fit["diagnostics"]["optimizer"]["converged"]
        forecast = kernel_predict_model(
            "complementary_goods", Dict("inputs"=>Dict("time"=>fixture["forecast_time"]), "state"=>fit["state"]),
        )["result"]
        @test maximum(abs.(forecast["rows"][1].-fixture["forecast_rows"][1])) < 1e-5
    end
end

@testset "Lock in forcing fit and history" begin
    fixture = Innovate.JSON.parsefile(
        joinpath(@__DIR__, "../../../rust/crates/innovate-core/tests/fixtures/lock_forcing_scipy.json");
        dicttype=Dict{String,Any},
    )
    start = copy(fixture["parameters"])
    start["alpha1"] = 0.29
    fit = kernel_fit_model("lock_in", Dict(
        "inputs"=>Dict("time"=>fixture["time"], "covariates"=>fixture["covariates"], "observed"=>fixture["rows"]),
        "constructor_kwargs"=>Dict("initial_values"=>fixture["initial"], "covariates"=>["price"]),
        "fit_options"=>Dict("initial_parameters"=>start, "max_iterations"=>200),
    ))["result"]
    @test fit["diagnostics"]["optimizer"]["converged"]
    @test fit["diagnostics"]["structural_identifiability"]["parameters"] == 14
    forecast = kernel_predict_model(
        "lock_in", Dict("inputs"=>Dict("time"=>fixture["forecast_time"]), "state"=>fit["state"]),
    )["result"]
    @test maximum(abs.(forecast["rows"][1].-fixture["forecast_rows"][1])) < 1e-3

    parameters = Dict("alpha1"=>0., "alpha2"=>0., "beta1"=>0., "beta2"=>0., "gamma1"=>0., "gamma2"=>0., "m"=>100.)
    for name in collect(keys(parameters))
        parameters["beta_"*name*"_price"] = 0.
    end
    parameters["beta_m_price"] = -10.
    result = kernel_predict_model("lock_in", Dict(
        "inputs"=>Dict("time"=>[0.,1.,2.,3.], "covariates"=>Dict("price"=>[0.,1.,2.,0.])),
        "parameters"=>parameters,
        "constructor_kwargs"=>Dict("initial_values"=>[95.,0.], "covariates"=>["price"]),
    ))["result"]
    @test [row[1] for row in result["rows"]] == [95.,90.,80.,95.]
end

@testset "Fixed ecosystem parameter events" begin
    fixture = Innovate.JSON.parsefile(
        joinpath(@__DIR__, "../../../rust/crates/innovate-core/tests/fixtures/ecosystem_events_scipy.json");
        dicttype=Dict{String,Any},
    )
    for model in ["lotka_volterra", "complementary_goods", "lock_in", "multi_product"]
        case = fixture["cases"][model]
        result = kernel_predict_model(model, Dict(
            "inputs"=>Dict("time"=>fixture["time"], "covariates"=>fixture["covariates"]),
            "parameters"=>case["parameters"], "constructor_kwargs"=>case["constructor_kwargs"],
        ))["result"]
        @test maximum(abs.(result["rows"][end].-case["rows"][end])) < 1e-5
        future = kernel_predict_model(
            model, Dict("inputs"=>Dict("time"=>fixture["forecast_time"]), "state"=>result["state"]),
        )["result"]
        @test maximum(abs.(future["rows"][1].-case["forecast_rows"][1])) < 1e-5
    end
    case = fixture["cases"]["lotka_volterra"]
    start = copy(case["parameters"])
    start["alpha1"] = 0.29
    fit = kernel_fit_model("lotka_volterra", Dict(
        "inputs"=>Dict("time"=>fixture["time"], "covariates"=>fixture["covariates"], "observed"=>case["rows"]),
        "constructor_kwargs"=>case["constructor_kwargs"],
        "fit_options"=>Dict("initial_parameters"=>start, "max_iterations"=>200),
    ))["result"]
    @test fit["diagnostics"]["optimizer"]["converged"]
    @test !fit["diagnostics"]["parameter_events"]["parameters_estimated"]
end

@testset "Norton fixed event variants" begin
    for fixture_name in ["norton_events_scipy.json", "successive_forcing_scipy.json"], explicit_start in (true, false)
        fixture = Innovate.JSON.parsefile(
            joinpath(@__DIR__, "../../../rust/crates/innovate-core/tests/fixtures", fixture_name);
            dicttype=Dict{String,Any},
        )
        options = Dict{String,Any}("max_iterations"=>200)
        if explicit_start
            start = copy(fixture["parameters"])
            start["p1"] = 0.0305
            options["initial_parameters"] = start
        end
        fit = kernel_fit_model("norton_bass", Dict(
            "inputs"=>Dict("time"=>fixture["time"], "covariates"=>fixture["covariates"], "observed"=>fixture["rows"]),
            "constructor_kwargs"=>fixture["constructor_kwargs"],
            "fit_options"=>options,
        ))["result"]
        @test fit["diagnostics"]["optimizer"]["converged"]
        @test !fit["diagnostics"]["parameter_events"]["parameters_estimated"]
        forecast = kernel_predict_model(
            "norton_bass", Dict("inputs"=>Dict("time"=>fixture["forecast_time"]), "state"=>fit["state"]),
        )["result"]
        @test maximum(abs.(forecast["rows"][1].-fixture["forecast_rows"][1])) < 1e-3
    end
end

@testset "Norton baseline threshold summary" begin
    result = kernel_summarize_model("norton_bass", Dict(
        "inputs"=>Dict("time"=>[4.]),
        "parameters"=>Dict("p1"=>0.1,"q1"=>0.9,"m1"=>100.),
        "constructor_kwargs"=>Dict("variant"=>"successive_generations","launch_times"=>[4.]),
    ))["result"]
    row = only(result["threshold_diagnostics"]["generations"])
    @test row["reference_time"] == 4.
    @test row["half_life"] ≈ log(9)
    @test row["saturation"] ≈ 2log(9)
end

@testset "Norton forced rate peak" begin
    result = kernel_summarize_model("norton_bass", Dict(
        "inputs"=>Dict("time"=>[0.,1.,2.,3.]),
        "parameters"=>Dict("p1"=>0.1,"q1"=>0.,"m1"=>100.),
        "constructor_kwargs"=>Dict("variant"=>"successive_generations",
            "parameter_events"=>[Dict("time"=>2.,"parameters"=>Dict("p1"=>0.5))]),
    ))["result"]
    trajectory = result["threshold_diagnostics"]["trajectory"]
    row = only(trajectory["generations"])
    @test trajectory["window"] == [0.,3.]
    @test row["peak_time"] == 2.
    @test row["peak_rate"] ≈ 50exp(-0.2)
    milestones = only(result["threshold_diagnostics"]["intrinsic_adoption_milestones"]["generations"])
    @test milestones["median"] ≈ 2 + (log(2) - 0.2) / 0.5
    @test milestones["saturation"] === nothing
end

@testset "Norton historical level milestones" begin
    result = kernel_summarize_model("norton_bass", Dict(
        "inputs"=>Dict("time"=>[1.,2.,3.]),
        "parameters"=>Dict("p1"=>0.1,"q1"=>0.,"m1"=>100.),
        "constructor_kwargs"=>Dict("initial_time"=>1.,"initial_values"=>[12.],
            "parameter_events"=>[Dict("time"=>2.,"parameters"=>Dict("m1"=>1000.))]),
    ))["result"]
    row = only(result["threshold_diagnostics"]["net_level_milestones"]["generations"])
    @test row["reference_market"] == 100.
    @test [row[k] for k in ("takeoff", "median", "saturation")] == [1.,3.,3.]
end

@testset "Scheduled Kairos agent hazards" begin
    options = Dict("engine"=>"kairos_network","population"=>2,"seed"=>42,"edges"=>[],
        "capacity"=>1,"service_ticks"=>4,"ticks_per_unit"=>1,"innovation_multipliers"=>[1,0],
        "schedule"=>[Dict("tick"=>2,"innovation_multipliers"=>[0,1])])
    result = kernel_simulate_model("bass", Dict("inputs"=>Dict("time"=>[0,2,5,8]),
        "parameters"=>Dict("p"=>1e12,"q"=>0,"m"=>2),"simulation_options"=>options))["result"]
    @test result["requests"] == [Dict("tick"=>1,"agent"=>0),Dict("tick"=>3,"agent"=>1)]
    @test first(result["completions"]) == Dict("tick"=>5,"agent"=>0)
end

@testset "Continuous Kairos hazard wakeup" begin
    forcing = Dict("time"=>[0,2,3],"innovation"=>[0,0,1],"imitation"=>[1,1,1])
    options = Dict("engine"=>"kairos_network","population"=>1,"seed"=>42,"edges"=>[],
        "capacity"=>1,"service_ticks"=>1,"ticks_per_unit"=>1,"hazard_forcing"=>forcing)
    result = kernel_simulate_model("bass", Dict("inputs"=>Dict("time"=>[0,2,3,4]),
        "parameters"=>Dict("p"=>1e12,"q"=>0,"m"=>1),"simulation_options"=>options))["result"]
    @test result["requests"] == [Dict("tick"=>3,"agent"=>0)]
    @test result["counts"] == [0,0,0,1]
end

@testset "Compensated rolling average" begin
    @test kernel_rolling_average([1e100, 1., 3.], 2) == [nothing, 5e99, 2.]
end

@testset "Dataset ingestion diagnostics" begin
    dataset = Dict("kind"=>"adoption", "time"=>[0.,1.], "adoption"=>[1.,2.],
        "provenance"=>Dict("source"=>"synthetic", "license"=>"CC-BY-4.0",
            "extraction_time"=>"2026-11-07T00:00:00Z"))
    result = kernel_call("ingest_dataset", nothing, Dict("dataset"=>dataset))["result"]
    @test result["validation"]["ok"]
    @test kernel_call("ingest_dataset", nothing, Dict("dataset"=>result["dataset"]))["result"] == result
    @test kernel_call("ingest_dataset", nothing, Dict("artifact"=>result))["result"] == result
    modified = deepcopy(result)
    modified["dataset"]["adoption"][2] = 3.
    @test_throws r"checksum does not match" kernel_call("ingest_dataset", nothing, Dict("artifact"=>modified))
    dataset["adoption"] = [1.,0.]
    error = try
        kernel_call("ingest_dataset", nothing, Dict("dataset"=>dataset))
        nothing
    catch exception
        exception
    end
    @test error isa DatasetValidationError
    @test error.code == "invalid_request"
    @test error.validation["failed"] == ["monotonicity:adoption"]
end

@testset "Offset timestamp aggregation" begin
    dates = ["2026-11-08T10:00:00+10:00", "2026-11-01T00:00:00Z", "2026-11-07T23:59:59.999Z"]
    result = kernel_aggregate_dates(dates, [4.,1.,2.], "W")
    @test result["dates"] == ["2026-11-01T00:00:00Z", "2026-11-08T00:00:00Z"]
    @test result["values"] == [3.,4.]
end

@testset "Process-local extension registry" begin
    manifest = Dict("name"=>"binding-fixture", "version"=>"1", "entrypoint"=>"demo:register",
        "stability"=>"beta", "extension_points"=>["diagnostics"])
    registered = kernel_call("register_extension_manifest", nothing, Dict("manifest"=>manifest))["result"]["manifest"]
    @test registered["stability"] == "provisional"
    @test kernel_call("list_extension_manifests")["result"]["manifests"] == [registered]
end

@testset "Paired Arrow validation" begin
    params = [Dict("name"=>name, "unit"=>unit, "value"=>value)
              for (name, unit, value) in [("p","period^-1",0.03), ("q","period^-1",0.3), ("m","people",100.0)]]
    strategies = [Dict("scenario_id"=>id, "time_unit"=>"period", "provenance"=>"synthetic",
        "draws"=>Any[Dict("draw_id"=>1, "parameters"=>params, "time"=>[0,1], "adoption"=>[0,1])])
        for id in ["control","policy"]]
    payload = Dict("strategies"=>strategies)
    source = kernel_paired_outcomes_ipc(payload)
    @test String(source[1:6]) == "ARROW1"
    portable = export_ipc_file(source)
    @test portable[1:12] == UInt8[0x41,0x52,0x52,0x4f,0x57,0x31,0,0,0xff,0xff,0xff,0xff]
    @test export_ipc_file(portable) == portable
    @test_throws ErrorException export_ipc_file(UInt8[0x41,0x52])
    draw = strategies[1]["draws"][1]
    strategies[1]["draws"][1] = [draw[key] for key in ["draw_id","parameters","time","adoption"]]
    @test_throws r"must be objects" kernel_paired_outcomes_ipc(payload)
end

@testset "Kairos resource comparisons" begin
    fixture = joinpath(@__DIR__, "../../../rust/crates/innovate-core/tests/fixtures/kairos_resource_comparisons.json")
    payload = Innovate.JSON.parsefile(fixture)
    response = kernel_call("execute_scenario_grid", nothing, payload)
    result = response["result"]
    @test length(result["executions"]) == 3
    success, failure = result["comparisons"]
    totals = success["result"]["resource_differences"]
    @test totals["configured_slot_ticks"] == -3
    @test totals["peak_in_service"] == 1
    @test totals["horizon_ticks"] == 11
    @test totals["ticks_per_unit"] == 1
    @test failure["error"]["code"] == "invalid_comparison"
    @test isnothing(failure["result"])

    for scenario in payload["scenarios"]
        scenario["simulation"]["inputs"]["time"] = [0, 7]
    end
    shortened = kernel_call("execute_scenario_grid", nothing, payload)
    totals = shortened["result"]["comparisons"][1]["result"]["resource_differences"]
    @test totals["waiting_at_horizon"] == 1
    @test totals["in_service_at_horizon"] == 2
end

@testset "Network ingestion retains Kairos compatibility warnings" begin
    dataset = Dict("kind"=>"network_edges", "source"=>["a","a"],
        "target"=>["a","b"], "weight"=>[1.0,0.0],
        "provenance"=>Dict("source"=>"synthetic", "license"=>"CC-BY-4.0",
            "extraction_time"=>"2026-11-07T00:00:00Z"))
    artifact = kernel_call("ingest_dataset", nothing, Dict("dataset"=>dataset))["result"]
    @test artifact["dataset"]["weight"] == [1.0,0.0]
    @test artifact["validation"]["warnings"] == ["self_loops:kairos_network","zero_weights:kairos_network"]
    @test kernel_call("ingest_dataset", nothing, Dict("artifact"=>artifact))["result"] == artifact
end

@testset "Stratified treatment contrast" begin
    for (estimand, estimate, weight) in [("ate", 6, 3/7), ("att", 7.25, 1/4)]
        payload = Dict("method"=>"stratified_treatment_contrast",
            "outcomes"=>[0,0,2,10,19,19,19], "treatment"=>[0,0,1,0,1,1,1],
            "strata"=>vcat(fill("low",3), fill("high",4)))
        payload["estimand"] = estimand
        result = kernel_call("summarize_causal", nothing, payload)["result"]
        @test result["estimate"] ≈ estimate
        @test result["by_stratum"]["low"]["weight"] ≈ weight
        @test result["by_stratum"]["low"]["treated_fraction"] ≈ 1/3
        if estimand == "att"
            with_control_only = merge(payload, Dict("strata"=>vcat("control_only", payload["strata"][2:end])))
            overlap = kernel_call("summarize_causal", nothing, with_control_only)["result"]
            @test overlap["excluded_control_observations"] == 1
            @test overlap["estimate"] ≈ estimate
        end
        @test haskey(result, "conditional_standard_error") && isnothing(result["conditional_standard_error"])
        replicated = merge(payload, Dict("outcomes"=>[0,2,4,6,10,10,12,12],
            "treatment"=>repeat([0,0,1,1],2), "strata"=>repeat(["low","high"],inner=4)))
        result = kernel_call("summarize_causal", nothing, replicated)["result"]
        @test result["conditional_standard_error"] ≈ sqrt(0.5)
        payload["strata"][end] = "unsupported"
        @test_throws r"each stratum requires treated and control observations" kernel_call("summarize_causal", nothing, payload)
    end
end

@testset "Treatment contrast discovery" begin
    result = kernel_call("summarize_causal", nothing, Dict("method"=>"describe_treatment_contrasts"))["result"]
    descriptive, adjusted, weighted, matching = result["contrasts"][1:4]
    @test descriptive["adjustment"] == "none"
    @test adjusted["required_fields"] == ["outcomes", "treatment", "strata"]
    @test adjusted["estimands"] == ["ate", "att"]
    @test matching["estimands"] == ["att"]
    @test matching["required_fields"] == ["outcomes", "treatment", "scores", "caliper"]
    @test weighted["required_fields"] == ["outcomes", "treatment", "propensities"]
    payload = Dict("method"=>weighted["method"], "outcomes"=>[0,10,2,12],
        "treatment"=>[0,0,1,1], "propensities"=>[0.2,0.8,0.2,0.8])
    result = kernel_call("summarize_causal", nothing, payload)["result"]
    @test result["estimate"] ≈ -4
    @test result["fixed_propensity_standard_error"] ≈ sqrt(20.48)
    singleton = merge(payload, Dict("outcomes"=>[0,2], "treatment"=>[0,1], "propensities"=>[0.5,0.5]))
    @test isnothing(kernel_call("summarize_causal", nothing, singleton)["result"]["fixed_propensity_standard_error"])
    for arm in ("treated", "control")
        diagnostics = result["weight_diagnostics"][arm]
        @test diagnostics["observations"] == 2
        @test diagnostics["maximum_normalized_weight"] ≈ 0.8
        @test diagnostics["effective_sample_size"] ≈ 25/17
    end
    @test weighted["estimands"] == ["ate", "att"]
    att = kernel_call("summarize_causal", nothing, merge(payload, Dict("estimand"=>"att")))["result"]
    @test att["estimate"] ≈ -41/17
    @test att["weight_diagnostics"]["treated"]["maximum_normalized_weight"] == 0.5
    payload["propensities"][1] = 0
    @test_throws r"strictly between zero and one" kernel_call("summarize_causal", nothing, payload)
end

@testset "Fitted stratum propensity workflow" begin
    treatment = [0,1,0,1,0,1,1]
    fit = kernel_call("summarize_causal", nothing, Dict("method"=>"estimate_stratum_propensities",
        "treatment"=>treatment, "strata"=>["low","high","low","low","high","high","high"]))["result"]
    @test fit["propensities"] ≈ [1/3,0.75,1/3,1/3,0.75,0.75,0.75]
    weighted = kernel_call("summarize_causal", nothing, Dict("method"=>"propensity_weighted_contrast",
        "outcomes"=>[0,19,0,2,10,19,19], "treatment"=>treatment, "propensities"=>fit["propensities"]))["result"]
    @test weighted["estimate"] ≈ 6
end

@testset "Nearest-score matching" begin
    methods = kernel_call("summarize_causal", nothing, Dict("method"=>"describe_treatment_contrasts"))["result"]["contrasts"]
    matching = only(filter(method -> method["adjustment"] == "nearest_score_with_replacement", methods))
    payload = Dict("method"=>matching["method"], "outcomes"=>[0,10,12,3,99],
        "treatment"=>[0,0,1,1,0], "scores"=>[0,2,2,1,0], "caliper"=>1)
    result = kernel_call("summarize_causal", nothing, payload)["result"]
    @test result["estimate"] ≈ 2.5
    @test [pair["control_index"] for pair in result["pairs"]] == [1,0]
    @test result["control_diagnostics"] == Dict("unique_controls"=>2, "effective_sample_size"=>2)
    @test isnothing(result["conditional_standard_error"])
    reused = kernel_call("summarize_causal", nothing,
        merge(payload, Dict("scores"=>[0,2,2,2,0], "outcome_variances"=>[1,4,1,1,1])))["result"]
    @test reused["conditional_standard_error"] ≈ sqrt(4.5)
    @test reused["control_diagnostics"] == Dict("unique_controls"=>1, "effective_sample_size"=>1)
    payload["caliper"] = 0.5
    partial = kernel_call("summarize_causal", nothing, merge(payload, Dict("allow_partial"=>true)))["result"]
    @test partial["estimand"] == "matched_subset_att"
    @test partial["estimate"] ≈ 2
    @test partial["unmatched_treated_indices"] == [3]
    @test_throws r"requires a control within the caliper" kernel_call("summarize_causal", nothing, payload)
end

@testset "Ridge logistic propensity fit" begin
    fit = kernel_call("summarize_causal", nothing, Dict("method"=>"fit_logistic_propensities",
        "treatment"=>[0,1], "covariates"=>[[-2],[2]], "ridge"=>0.25/log(3)))["result"]
    @test isapprox(fit["propensities"], [0.25,0.75]; atol=1e-7)
    @test fit["covariate_scales"] == [2]
    prediction = kernel_call("summarize_causal", nothing, Dict("method"=>"predict_logistic_propensities",
        "fit"=>fit, "covariates"=>[[0],[2],[-2]]))["result"]
    @test isapprox(prediction["propensities"], [0.5,0.75,0.25]; atol=1e-7)
    score = kernel_call("summarize_causal", nothing, Dict("method"=>"score_propensities",
        "treatment"=>[0,1,0], "propensities"=>prediction["propensities"]))["result"]
    @test isapprox(score["brier_score"], 0.125; atol=1e-7)
    @test isapprox(score["log_loss"], -(log(0.5) + 2 * log(0.75)) / 3; atol=1e-7)
    cross_fit = kernel_call("summarize_causal", nothing, Dict("method"=>"cross_fit_logistic_propensities",
        "treatment"=>[0,1,1,0], "covariates"=>[[-2],[1],[2],[-1]], "folds"=>[9,4,9,4], "ridge"=>0.25/log(3)))["result"]
    low = 1 / (1 + sqrt(3))
    @test isapprox(cross_fit["propensities"], [0.1,1-low,0.9,low]; atol=1e-7)
end

@testset "Augmented treatment contrast" begin
    result = kernel_call("summarize_causal", nothing, Dict("method"=>"augmented_treatment_contrast",
        "outcomes"=>[0,4], "treatment"=>[0,1], "propensities"=>[0.5,0.5],
        "treated_predictions"=>[2,2], "control_predictions"=>[0,0]))["result"]
    @test result["estimate"] ≈ 4
    @test result["influence_standard_error"] ≈ 2
    att = kernel_call("summarize_causal", nothing, Dict("method"=>"augmented_treatment_contrast",
        "estimand"=>"att", "outcomes"=>[1,4], "treatment"=>[0,1], "propensities"=>[0.5,0.5],
        "control_predictions"=>[0,1]))["result"]
    @test att["estimand"] == "att"
    @test att["estimate"] ≈ 2
    @test att["influence_standard_error"] ≈ 2
end

@testset "Cross-fitted stratum ATE" begin
    result = kernel_call("summarize_causal", nothing, Dict("method"=>"cross_fitted_stratum_ate",
        "outcomes"=>[0,4,2,6], "treatment"=>[0,1,0,1], "strata"=>fill("a",4), "folds"=>[1,2,2,1]))["result"]
    @test result["control_predictions"] == [2,0,0,2]
    @test result["treated_predictions"] == [4,6,6,4]
    payload = Dict("method"=>"cross_fitted_stratum_ate", "outcomes"=>[0,4,2,6],
        "treatment"=>[0,1,0,1], "strata"=>fill("a",4), "folds"=>[1,2,2,1], "clusters"=>["a","b","b","a"])
    clustered = kernel_call("summarize_causal", nothing, payload)["result"]["contrast"]
    @test clustered["cluster_count"] == 2
    @test clustered["influence_standard_error"] ≈ 2
    att = kernel_call("summarize_causal", nothing,
        merge(payload, Dict("method"=>"cross_fitted_stratum_att")))["result"]["contrast"]
    @test att["estimand"] == "att"
    @test att["estimate"] ≈ 4
    @test att["cluster_count"] == 2
    @test att["influence_standard_error"] ≈ 2
    for method in ("cross_fitted_stratum_ate", "cross_fitted_stratum_att")
        request = merge(payload, Dict("method"=>method, "minimum_training_arm_clusters"=>1))
        guarded = kernel_call("summarize_causal", nothing, request)["result"]
        @test guarded["overlap"]["min_training_arm_clusters"] == 1
        @test guarded["contrast"]["cluster_degrees_of_freedom"] == 1
        request["minimum_training_arm_clusters"] = 2
        @test_throws r"falls below minimum_training_arm_clusters" kernel_call("summarize_causal", nothing, request)
    end
    payload["clusters"] = ["a","a","b","b"]
    @test_throws r"single cross-fitting fold" kernel_call("summarize_causal", nothing, payload)
    @test result["contrast"]["estimate"] ≈ 4
end

@testset "Cross-fitted stratum estimands" begin
    payload = Dict{String,Any}("outcomes"=>repeat([0,0,2,10,19,19,19],2),
        "treatment"=>repeat([0,0,1,0,1,1,1],2), "strata"=>repeat(vcat(fill("low",3),fill("high",4)),2),
        "folds"=>repeat([1,2],inner=7))
    for (estimand, estimate) in [("ate",6), ("att",7.25)]
        payload["method"] = "cross_fitted_stratum_" * estimand
        result = kernel_call("summarize_causal", nothing, payload)["result"]["contrast"]
        @test result["estimand"] == estimand
        @test result["estimate"] ≈ estimate
    end
end

@testset "Clustered augmented contrast" begin
    payload = Dict("method"=>"augmented_treatment_contrast", "outcomes"=>[0,4,0,4],
        "treatment"=>[0,1,0,1], "propensities"=>fill(0.5,4),
        "treated_predictions"=>fill(2,4), "control_predictions"=>fill(0,4))
    independent = kernel_call("summarize_causal", nothing, payload)["result"]
    clustered = kernel_call("summarize_causal", nothing, merge(payload, Dict("clusters"=>["a","b","a","b"])))["result"]
    @test clustered["estimate"] == independent["estimate"] == 4
    @test clustered["cluster_count"] == 2
    @test clustered["influence_standard_error"] ≈ 2
end

@testset "Balanced-panel event study" begin
    payload = Dict("method"=>"panel_event_study", "periods"=>[-2,-1,0], "baseline_index"=>1,
        "treated"=>[[1,2,7],[3,4,11]], "control"=>[[0,1,3],[4,5,7]])
    pre, reference, post = kernel_call("summarize_causal", nothing, payload)["result"]["contrasts"]
    @test pre["estimate"] == 0
    @test reference["reference"]
    @test post["estimate"] ≈ 4
    @test post["standard_error"] ≈ 1
end

@testset "Staggered panel ATT" begin
    payload = Dict("method"=>"staggered_panel_att", "periods"=>[0,1,2],
        "outcomes"=>[[0,5,7],[10,11,16]], "adoption_indices"=>[1,2])
    result = kernel_call("summarize_causal", nothing, payload)["result"]
    first, second, third = result["contrasts"]
    @test result["event_time"][1]["unsupported_cohorts"] == 1
    @test isnothing(result["event_time"][2]["supported_cohort_att"])
    @test first["estimate"] ≈ 4
    @test first["n_control"] == 1
    @test all(row -> isnothing(row["estimate"]) && row["n_control"] == 0, (second,third))
    @test isnothing(first["standard_error"])
    replicated = merge(payload, Dict("outcomes"=>[[0,5,7],[0,7,9],[10,12,16],[10,12,16]],
        "adoption_indices"=>[1,1,2,nothing]))
    result = kernel_call("summarize_causal", nothing, replicated)["result"]
    rows = result["contrasts"]
    @test result["event_time"][1]["supported_cohort_att"] ≈ 8/3
    @test isnothing(result["event_time"][1]["standard_error"])
    shared = merge(payload, Dict("outcomes"=>[[0,5,5],[0,5,5],[0,1,6],[0,1,6],[0,0,0],[0,2,4]],
        "adoption_indices"=>[1,1,2,2,nothing,nothing]))
    joint = kernel_call("summarize_causal", nothing, merge(shared, Dict("include_event_time_covariance"=>true, "simultaneous_bands"=>Dict("draws"=>100,"seed"=>7))))["result"]
    onset = joint["event_time"][1]
    error = onset["standard_error"]
    @test joint["event_time_covariance"][1] ≈ [error^2,2*error]
    @test joint["event_time_covariance"][2] ≈ [2*error,4]
    bands = joint["simultaneous_bands"]
    @test bands["critical_value"] ≈ sqrt(2)
    @test bands["intervals"][2]["interval"]["lower"] ≈ 3 - 2*sqrt(2)
    @test bands["intervals"][2]["interval"]["upper"] ≈ 3 + 2*sqrt(2)
    @test onset["standard_error"] ≈ (1/sqrt(48) + 1/sqrt(8)) * sqrt(2)
    balanced = kernel_call("summarize_causal", nothing, merge(replicated, Dict("balanced_horizon"=>1)))["result"]
    @test balanced["balanced_event_time"]["excluded_cohorts"] == 1
    @test [row["supported_cohort_att"] for row in balanced["event_time"]] ≈ [4,2]
    @test balanced["contrasts"] == rows
    placebo = merge(payload, Dict("outcomes"=>[[0,5,7],[10,11,16],[0,2,4]],
        "adoption_indices"=>[1,2,nothing], "include_pre_trends"=>true))
    pre = only(kernel_call("summarize_causal", nothing, placebo)["result"]["pre_trends"])
    @test pre["estimate"] ≈ 1
    @test pre["n_control"] == 1
    @test pre["baseline_period"] == 1
    for (group, count, estimate) in [("not_yet_treated",2,3.5), ("never_treated",1,3)]
        onset = kernel_call("summarize_causal", nothing, merge(placebo, Dict("control_group"=>group)))["result"]["contrasts"][1]
        @test onset["n_control"] == count
        @test onset["estimate"] ≈ estimate
    end
    @test rows[1]["estimate"] ≈ 4
    @test rows[1]["standard_error"] ≈ 1
    @test rows[2]["n_control"] == 1
end

@testset "Staggered long panel ATT" begin
    payload = Dict("method"=>"staggered_long_panel_att", "unit_ids"=>["z","a","z","a","a","z"],
        "time"=>[20,10,10,30,20,30], "outcomes"=>[11,0,10,7,5,16], "adoption_times"=>Dict("a"=>20,"z"=>30))
    result = kernel_call("summarize_causal", nothing, payload)["result"]
    @test result["unit_order"] == ["a","z"]
    @test result["contrasts"][1]["estimate"] ≈ 4
end

@testset "Scenario grid ranking" begin
    scenarios = [Dict("id"=>id, "model_key"=>"logistic", "simulation"=>Dict(
        "inputs"=>Dict("time"=>[1,2,4]), "parameters"=>Dict("L"=>capacity,"k"=>1,"x0"=>1)))
        for (id,capacity) in [("base",100),("alt",200),("tie",200)]]
    result = kernel_call("execute_scenario_grid", nothing,
        Dict("scenarios"=>scenarios, "rank_scenarios"=>["tie","base","alt"]))["result"]
    @test [(row["id"],row["place"]) for row in result["ranking"]["rows"]] == [("tie",1),("alt",1),("base",3)]
    area = kernel_call("execute_scenario_grid", nothing, Dict("scenarios"=>scenarios,
        "rank_scenarios"=>["tie","base","alt"], "rank_metric"=>"adoption_area"))["result"]
    first, second, third = area["executions"][2]["result"]["values"]
    @test area["ranking"]["metric"] == "adoption_area"
    @test area["ranking"]["rows"][1]["adoption_area"] ≈ (first + second)/2 + second + third
    threshold = kernel_call("execute_scenario_grid", nothing, Dict("scenarios"=>scenarios,
        "rank_scenarios"=>["tie","base","alt"], "rank_metric"=>"time_to_threshold", "rank_threshold"=>150))["result"]["ranking"]["rows"]
    @test [(row["id"],row["place"]) for row in threshold] == [("tie",1),("alt",1),("base",nothing)]
    @test threshold[1]["time_to_threshold"] ≈ 2 + 2*(150-second)/(third-second)
    @test isnothing(threshold[3]["time_to_threshold"])
    push!(scenarios, merge(scenarios[1], Dict("id"=>"base2")))
    comparison = Dict("pairs"=>[Dict("baseline"=>"base","alternative"=>"alt"),
        Dict("baseline"=>"tie","alternative"=>"base2")], "interval_mass"=>0.5, "evaluation_metric"=>"adoption_area")
    paired = kernel_call("execute_scenario_grid", nothing, Dict("scenarios"=>scenarios,
        "comparisons"=>[comparison]))["result"]["comparisons"][1]["result"]
    expected = area["ranking"]["rows"][1]["adoption_area"]/2
    @test paired["evaluation_horizon"] == [1,4]
    @test paired["differences"] ≈ [expected,-expected]
    @test paired["mean_difference"] == 0
    delete!(comparison, "evaluation_metric")
    comparison["include_path_covariance"] = true
    endpoint = kernel_call("execute_scenario_grid", nothing, Dict("scenarios"=>scenarios,
        "comparisons"=>[comparison]))["result"]["comparisons"][1]["result"]
    rows = endpoint["path_summary"]["rows"]
    @test [row["time"] for row in rows] == [1,2,4]
    for field in ("mean_difference","mean_standard_error","empirical_interval")
        @test rows[end][field] == endpoint[field]
    end
    errors = [row["mean_standard_error"] for row in rows]
    for (row, error) in zip(endpoint["path_covariance"], errors)
        @test row ≈ error .* errors
    end

    delete!(comparison, "include_path_covariance")
    comparison["interval_mass"] = 0.9
    comparison["exact_sign_flip"] = true
    comparison["bootstrap"] = Dict("draws"=>1000, "seed"=>42)
    inference = kernel_call("execute_scenario_grid", nothing, Dict("scenarios"=>scenarios,
        "comparisons"=>[comparison]))["result"]["comparisons"][1]["result"]
    @test inference["sign_flip"]["assignments"] == 4
    @test inference["sign_flip"]["extreme_assignments"] == 4
    @test inference["sign_flip"]["p_value"] == 1
    effect = endpoint["differences"][1]
    @test inference["bootstrap"]["mean_interval"] ≈ [-effect,effect]
    @test inference["bootstrap"]["seed"] == 42
end

@testset "Adopter classification on irregular grids" begin
    origin = 1.5
    largest = floatmax(Float64)
    for time in ([origin, nextfloat(origin), nextfloat(origin, 4)], [-largest, -largest / 2, largest])
        result = kernel_call("categorize_adopters", nothing,
            Dict("time"=>time, "adoption_rate"=>[1,2,1]))["result"]
        @test result["time"] == time
        @test result["category"] == ["Early Adopters", "Early Majority", "Laggards"]
    end
    for origin in (0, 1024)
        result = kernel_call("categorize_adopters", nothing,
            Dict("time"=>collect(origin:origin+3), "adoption_rate"=>[0,1,0,1]))["result"]
        @test result["category"] == ["Innovators", "Early Adopters", "Early Majority", "Late Majority"]
    end
end

@testset "Kairos tick checkpoint resume" begin
    fixture = joinpath(@__DIR__, "../../../rust/crates/innovate-core/tests/fixtures/kairos_network_checkpoint.json")
    for engine in ("kairos_des", "kairos_abm", "kairos_network")
        payload = Innovate.JSON.parsefile(fixture)
        if engine != "kairos_network"
            payload["simulation_options"] = Dict{String,Any}(
                "engine"=>engine, "seed"=>42, "population"=>6, "ticks_per_unit"=>1)
        end
        expected = kernel_simulate_model("bass", payload)["result"]
        options = payload["simulation_options"]
        for tick in (0, 4, 6)
            options["pause_at_tick"] = tick
            paused = kernel_simulate_model("bass", payload)["result"]
            @test paused["status"] == "paused"
            delete!(options, "pause_at_tick")
            options["resume_checkpoint"] = paused["checkpoint"]
            @test kernel_simulate_model("bass", payload)["result"] == expected
            delete!(options, "resume_checkpoint")
        end
    end
end

@testset "Narrow matched balance" begin
    lower = 1.5
    upper = nextfloat(lower)
    payload = Dict("method"=>"matched_treatment_contrast", "outcomes"=>[0,0,1,1,1],
        "treatment"=>[0,0,1,1,1], "scores"=>[0,1,0,0,1], "caliper"=>0,
        "balance_covariates"=>Dict("narrow"=>[lower,upper,lower,upper,lower], "constant"=>ones(5)))
    balance = kernel_call("summarize_causal", nothing, payload)["result"]["covariate_balance"]
    @test balance["narrow"]["before_matching"]["standardized_mean_difference"] ≈ (-1/6)/sqrt(7/18)
    @test balance["narrow"]["standardized_mean_difference"] == 0
    @test balance["narrow"]["before_matching"]["mean_difference"] / (upper - lower) ≈ -1/6
    @test isnothing(balance["constant"]["standardized_mean_difference"])
end

@testset "Matching with cross-fitted controls" begin
    observations = Dict("outcomes"=>[0,4,2,6], "treatment"=>[0,1,0,1])
    fit = kernel_call("summarize_causal", nothing, merge(observations, Dict(
        "method"=>"cross_fitted_stratum_ate", "strata"=>fill("a",4), "folds"=>[1,2,2,1])))["result"]
    matched = kernel_call("summarize_causal", nothing, merge(observations, Dict(
        "method"=>"matched_treatment_contrast", "scores"=>[0,0,1,0], "caliper"=>0,
        "control_predictions"=>fit["control_predictions"])))["result"]
    @test matched["estimate"] == 5
    @test matched["adjusted_estimate"] == 6
    @test isnothing(matched["conditional_standard_error"])
end

@testset "Correlated matched uncertainty" begin
    payload = Dict("method"=>"matched_treatment_contrast", "outcomes"=>[0,10,12,3,99],
        "treatment"=>[0,0,1,1,0], "scores"=>[0,2,2,2,0], "caliper"=>1,
        "outcome_variances"=>[1,4,1,1,1], "outcome_covariance_factors"=>[fill(7,5),[0,0,2,2,0]])
    matched = kernel_call("summarize_causal", nothing, payload)["result"]
    @test matched["conditional_standard_error"] ≈ sqrt(8.5)
    @test matched["uncertainty_model"] == Dict("model"=>"diagonal_plus_factors", "factor_count"=>2)
end

@testset "Fitted matching covariance" begin
    result = kernel_call("summarize_causal", nothing, Dict(
        "method"=>"matched_treatment_contrast", "outcomes"=>[0,4,2,6],
        "treatment"=>[0,1,0,1], "scores"=>[0,0,1,0], "caliper"=>0,
        "outcome_variances"=>ones(4),
        "outcome_covariance_factors"=>[fill(7,4), [0,2,0,2]],
        "fit_controls"=>Dict("strata"=>fill("a",4), "folds"=>[1,2,2,1])))["result"]
    @test result["estimate"] == 5
    @test result["adjusted_estimate"] == 6
    @test result["adjusted_outcome_weights"] == [[0,-1.5],[1,0.5],[2,0.5],[3,0.5]]
    @test result["conditional_standard_error"] ≈ sqrt(5.5)
    @test result["adjusted_standard_error"] ≈ sqrt(7)
end

@testset "Matching with fitted scores" begin
    result = kernel_call("summarize_causal", nothing, Dict(
        "method"=>"matched_treatment_contrast", "outcomes"=>[0,4,2,6],
        "treatment"=>[0,1,0,1], "caliper"=>0,
        "fit_scores"=>Dict("covariates"=>[[0] for _ in 1:4], "folds"=>[1,1,2,2],
            "ridge"=>1)))["result"]
    @test result["estimate"] == 5
    @test result["score_fit"]["propensities"] == fill(0.5,4)
    @test result["score_uncertainty_scope"] == "conditional_on_fitted_scores_and_pairs"
    @test result["match_diagnostics"]["minimum_matched_caliper_slack"] == 0
    @test result["match_diagnostics"]["minimum_runner_up_distance_gap"] == 0
end

@testset "Matching score draw mixture" begin
    sensitivity = kernel_call("summarize_causal", nothing, Dict(
        "method"=>"matched_treatment_contrast", "outcomes"=>[0,4,2],
        "treatment"=>[0,1,0], "scores"=>[0,0,1], "caliper"=>0,
        "score_draws"=>[[0,0,1],[1,0,0]], "outcome_variances"=>[0,1,6],
        "control_predictions"=>[0,0,2]
    ))["result"]["score_sensitivity"]
    @test [draw["estimate"] for draw in sensitivity["draws"]] == [4,2]
    @test sensitivity["same_treated_subset"]
    @test sensitivity["mixture_standard_deviation"] ≈ sqrt(5)
    @test sensitivity["adjusted_mixture_standard_deviation"] ≈ 2
end

@testset "Matching score refits" begin
    result = kernel_call("summarize_causal", nothing, Dict(
        "method"=>"matched_treatment_contrast", "outcomes"=>[0,4,2,6,8,12,10,14],
        "treatment"=>repeat([0,1],4), "caliper"=>1, "outcome_variances"=>ones(8),
        "fit_scores"=>Dict("covariates"=>[[0] for _ in 1:8],
            "folds"=>[1,1,1,1,2,2,2,2], "ridge"=>1),
        "score_refit_omit_indices"=>[0,4]
    ))["result"]["score_sensitivity"]
    @test result["omitted_training_indices"] == [0,4]
    @test [pair["control_index"] for pair in result["draws"][1]["pairs"]] == [0,0,4,4]
    @test result["mixture_standard_deviation"] === nothing
end

@testset "Matching retains insufficient score draws" begin
    sensitivity = kernel_call("summarize_causal", nothing, Dict(
        "method"=>"matched_treatment_contrast", "outcomes"=>[0,4], "treatment"=>[0,1],
        "scores"=>[0,0], "caliper"=>0, "score_draws"=>[[0,0],[0,1]], "outcome_variances"=>[1,1]
    ))["result"]["score_sensitivity"]
    @test [draw["status"] for draw in sensitivity["draws"]] == ["matched","insufficient_matches"]
    @test sensitivity["draws"][1]["estimate"] == 4
    @test sensitivity["mixture_standard_deviation"] === nothing
end

@testset "Matching score bootstrap" begin
    payload = Dict("method"=>"matched_treatment_contrast", "outcomes"=>[0,4,2,6,8,12,10,14],
        "treatment"=>repeat([0,1],4), "caliper"=>1, "outcome_variances"=>ones(8),
        "fit_scores"=>Dict("covariates"=>[[x] for x in [0,3,2,5,1,4,3,6]],
            "folds"=>[1,1,1,1,2,2,2,2], "ridge"=>1),
        "score_bootstrap"=>Dict("draws"=>8, "seed"=>17))
    run() = kernel_call("summarize_causal", nothing, payload)["result"]["score_sensitivity"]
    result = run()
    @test result == run()
    @test result["seed"] == 17 && length(result["draws"]) == 8
    @test result["mixture_standard_deviation"] > 0
end

@testset "Extension dataset provider" begin
    selection = Dict("name"=>"provider-fixture", "expected_version"=>"1", "extension_point"=>"dataset_provider")
    manifest = Dict("name"=>selection["name"], "version"=>"1", "entrypoint"=>"fixture:provide",
        "stability"=>"stable", "extension_points"=>["dataset_provider"])
    kernel_call("register_extension_manifest", nothing, Dict("manifest"=>manifest))
    try
        dataset = Dict("kind"=>"adoption", "time"=>[0,1], "adoption"=>[1,2],
            "provenance"=>Dict("source"=>"fixture", "license"=>"CC-BY-4.0",
                "extraction_time"=>"2026-11-07T00:00:00Z"))
        artifact = invoke_registered_extension((_, data) -> data, selection, dataset)
        @test artifact["validation"]["ok"]
        @test artifact["dataset"]["provenance"]["checksum"] ==
            "fada5d71f8981fa746be73b26de45ad470fa4e74f6a7d7a48234f4f50de0a76a"
        @test_throws ErrorException invoke_registered_extension((_, _) -> Dict("kind"=>"adoption"), selection, nothing)
    finally
        kernel_call("unregister_extension_manifest", nothing,
            Dict("name"=>selection["name"], "expected_version"=>"1"))
    end
end

@testset "Empirical prediction intervals" begin
    result = kernel_call("calibrate_prediction_intervals", nothing, Dict(
        "time" => [1, 2, 3], "observed" => [11, 18, 35],
        "predicted" => [10, 20, 30], "confidence" => 0.5, "holdout" => [0, 1, 0],
    ))["result"]
    @test result["lower"] == [8, 18, 28]
    @test result["upper"] == [12, 22, 32]
    @test result["diagnostics"]["holdout_coverage"] == 1.0
end

@testset "Split conformal prediction intervals" begin
    result = kernel_call("calibrate_prediction_intervals", nothing, Dict(
        "time" => [1, 2, 3, 4], "observed" => [11, 12, 13, 100],
        "predicted" => [10, 10, 10, 10], "confidence" => 0.5,
        "holdout" => [0, 0, 0, 1], "method" => "split_conformal",
    ))["result"]
    @test result["metadata"]["interval_half_width"] == 2.0
    @test result["diagnostics"]["holdout_coverage"] == 0.0
end

@testset "Policy specification rejects positional contracts" begin
    payload = Dict("method" => "policy_specification",
        "intervention" => Dict("name" => "Pilot", "timing" => "post", "comparator" => "control"),
        "causal_model" => ["Evaluation", "policy", "adoption", ["age"], nothing, Dict()])
    @test_throws r"invalid_request" kernel_call("summarize_causal", nothing, payload)
end

@testset "Weighted regime ensemble" begin
    payload = Dict("time" => [1, 2], "predictions" => Dict("a" => [1, 2], "b" => [3, 4]),
        "weights" => Dict("a" => 1, "b" => 3), "observed" => [2.5, 3.5])
    result = kernel_call("compose_regime_ensemble", nothing, payload)["result"]
    @test result["mean"] == [2.5, 3.5]
    @test result["diagnostics"]["rmse"] == 0
end

@testset "Diagnostics artifact tables" begin
    artifact = Dict("columns" => ["z", "a"], "rows" => [Dict("a" => 1, "z" => 2), Dict("a" => 3), false])
    single = Dict("columns" => ["value"], "rows" => [Dict("value" => 4), Dict()])
    payload = Dict("artifacts" => Dict("ordered" => artifact, "single" => single))
    tables = kernel_call("diagnostics_artifact_tables", nothing, payload)["result"]["tables"]
    @test tables["ordered"]["columns"] == ["z", "a"]
    @test tables["ordered"]["rows"] == [[2, 1], [nothing, 3], [nothing, nothing]]
    @test tables["single"]["rows"] == [[4], [nothing]]
end

@testset "Streaming update and capability discovery" begin
    capabilities = kernel_call("list_advanced_capabilities", nothing, Dict())["result"]["capabilities"]
    streaming = only(filter(item -> item["key"] == "streaming_update", capabilities))
    @test streaming["supports_incremental"]
    payload = Dict("previous_time" => [1, 2], "previous_observed" => [2, 4],
        "new_time" => [3], "new_observed" => [6])
    result = kernel_call(streaming["operation"], nothing, payload)["result"]
    @test result["mean"] == [2, 4, 6]
    @test result["diagnostics"]["growth_rate"] == 0.5
end

@testset "Residual SVG export" begin
    payload = Dict("time" => [1, 2, 3, 4], "residuals" => [-1, 1, -1, 1],
        "config" => Dict("lags" => 2, "acf_only" => true))
    result = kernel_call("render_residual_diagnostics_svg", nothing, payload)["result"]
    @test result["series"]["acf"] == [1, -0.75, 0.5]
    @test startswith(result["svg"], "<svg")
    @test !occursin("Partial autocorrelation", result["svg"])
end

if get(ENV, "INNOVATE_TEST_LIVE_OECD", "") == "1"
    @testset "Live OECD SDMX retrieval" begin
        payload = Dict("dataset_id" => "OECD.SDD.STES,DSD_STES@DF_CLI",
            "dimension_key" => ".M.LI...AA...H", "start_date" => "2023-02", "end_date" => "2023-03")
        result = kernel_call("fetch_oecd_dataset", nothing, payload)["result"]
        @test result["request"]["provider"] == "OECD"
        lines = split(result["csv"], '\n'; keepempty=false)
        @test occursin("TIME_PERIOD,OBS_VALUE", first(lines))
        @test any(line -> occursin(r",2023-0[23],", line), lines[2:end])
    end
end

@testset "Historical competition equilibrium" begin
    operation = "summarize_competition_equilibrium"
    shares = kernel_call(operation, "multi_product", Dict(
        "market_potentials" => [100, 300]))["result"]
    @test shares["shares"] == Dict("product_0" => 0.25, "product_1" => 0.75)
    @test !shares["interaction_adjusted"]
    boundary = kernel_call(operation, "lotka_volterra")["result"]
    @test (boundary["N1_eq"], boundary["N2_eq"]) == (1000, 0)
end

@testset "Historical competition cross-elasticity" begin
    time = [0, 1, 2]
    result = kernel_call("competition_cross_elasticity", nothing, Dict(
        "time" => time, "p" => [0.1, 0.2], "q" => [[0, 0.4], [0, 0]],
        "market_potentials" => [50, 100]))["result"]
    @test length(result["elasticity"]) == 2
    first, second = result["elasticity"]
    expected = 40 .* (1 .- exp.(-0.2 .* time)) ./ (50 + 1e-10)
    @test isapprox(first, expected; atol=1e-7, rtol=0)
    @test second == [0, 0, 0]
end
