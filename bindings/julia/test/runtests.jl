using Test, Innovate
@testset "Native conformance" begin
    payload = Dict("inputs" => Dict("time" => [0.,1.,2.]), "parameters" => Dict("L"=>100.,"k"=>1.,"x0"=>1.))
    result = kernel_predict_model("logistic", payload)
    @test result["metadata"]["runtime"] == "rust_native"
    @test result["result"]["values"][2] == 50.0
    @test length(kernel_discover_models()["models"]) == 5
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
