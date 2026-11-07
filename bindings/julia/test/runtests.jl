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
    payload=Dict("inputs"=>Dict("time"=>fixture["time"],"observed"=>fixture["rows"],"covariates"=>fixture["covariates"]),"constructor_kwargs"=>Dict("initial_values"=>fixture["initial"],"covariates"=>["price"]),"fit_options"=>Dict("initial_parameters"=>start,"max_iterations"=>150))
    fit=kernel_fit_model("lotka_volterra",payload)["result"]
    @test fit["diagnostics"]["optimizer"]["converged"]
    forecast=kernel_predict_model("lotka_volterra",Dict("inputs"=>Dict("time"=>fixture["forecast_time"]),"state"=>fit["state"]))["result"]
    @test maximum(abs.(forecast["rows"][1].-fixture["forecast_rows"][1]))<1e-5
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
    fit = kernel_fit_model("complementary_goods", Dict(
        "inputs"=>Dict("time"=>fixture["time"], "covariates"=>fixture["covariates"], "observed"=>fixture["rows"]),
        "constructor_kwargs"=>Dict("initial_values"=>fixture["initial"], "covariates"=>["price"]),
        "fit_options"=>Dict("initial_parameters"=>start, "max_iterations"=>150),
    ))["result"]
    @test fit["diagnostics"]["optimizer"]["converged"]
    forecast = kernel_predict_model(
        "complementary_goods", Dict("inputs"=>Dict("time"=>fixture["forecast_time"]), "state"=>fit["state"]),
    )["result"]
    @test maximum(abs.(forecast["rows"][1].-fixture["forecast_rows"][1])) < 1e-5
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
    for fixture_name in ["norton_events_scipy.json", "successive_forcing_scipy.json"]
        fixture = Innovate.JSON.parsefile(
            joinpath(@__DIR__, "../../../rust/crates/innovate-core/tests/fixtures", fixture_name);
            dicttype=Dict{String,Any},
        )
        start = copy(fixture["parameters"])
        start["p1"] = 0.0305
        fit = kernel_fit_model("norton_bass", Dict(
            "inputs"=>Dict("time"=>fixture["time"], "covariates"=>fixture["covariates"], "observed"=>fixture["rows"]),
            "constructor_kwargs"=>fixture["constructor_kwargs"],
            "fit_options"=>Dict("initial_parameters"=>start, "max_iterations"=>200),
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
end
