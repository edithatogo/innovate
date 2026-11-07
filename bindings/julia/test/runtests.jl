using Test, Innovate
@testset "Native conformance" begin
    payload = Dict("inputs" => Dict("time" => [0.,1.,2.]), "parameters" => Dict("L"=>100.,"k"=>1.,"x0"=>1.))
    result = kernel_predict_model("logistic", payload)
    @test result["metadata"]["runtime"] == "rust_native"
    @test result["result"]["values"][2] == 50.0
    @test length(kernel_discover_models()["models"]) == 7
    @test_throws ErrorException kernel_predict_model("latent_process", payload)
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
