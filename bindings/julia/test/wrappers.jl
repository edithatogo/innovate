using Test
using Innovate

time = [0.0, 1.0, 2.0, 3.0, 4.0]
observed = [0.02, 0.06, 0.12, 0.25, 0.41]

@testset "Julia binding wrappers" begin
    discovery = Innovate.kernel_discover_models()
    @test discovery isa AbstractVector
    @test !isempty(discovery)

    bass = first(filter(record -> record["key"] == "bass", discovery))
    @test bass["family"] == "diffusion"
    @test haskey(bass, "supports_summarize")

    fit_request = Innovate.kernel_request(
        operation = "fit_model",
        model_key = bass["key"],
        payload = Dict(
            "inputs" => Dict("time" => time, "observed" => observed),
            "model_kwargs" => Dict{String,Any}(),
        ),
    )

    fit = Innovate.kernel_fit_model(fit_request)
    @test fit isa AbstractDict
    @test fit["model_key"] == "bass"
    @test haskey(fit, "predictions")
    @test haskey(fit, "diagnostics")

    diagnostics = Innovate.kernel_extract_diagnostics(fit)
    @test diagnostics isa AbstractDict
    @test diagnostics["support_level"] == "supported"

    predict = Innovate.kernel_predict_model(
        Innovate.kernel_request(
            operation = "predict_model",
            model_key = bass["key"],
            payload = Dict(
                "inputs" => Dict("time" => time),
                "state" => fit["state"],
            ),
        ),
    )
    @test predict isa AbstractArray || predict isa AbstractVector

    summarize = Innovate.kernel_summarize_model(
        Innovate.kernel_request(
            operation = "summarize_model",
            model_key = bass["key"],
            payload = Dict(
                "inputs" => Dict("time" => time, "observed" => observed),
                "state" => fit["state"],
            ),
        ),
    )
    @test summarize["state"]["model_key"] == "bass"
    @test haskey(summarize, "diagnostics")

    diagnose = Innovate.kernel_diagnose_model(
        Innovate.kernel_request(
            operation = "diagnose_model",
            model_key = bass["key"],
            payload = Dict(
                "inputs" => Dict("time" => time, "observed" => observed),
                "state" => fit["state"],
            ),
        ),
    )
    @test diagnose["diagnostics"]["support_level"] == "supported"
    @test haskey(diagnose, "state")

    bad_request = Innovate.kernel_request(
        operation = "fit_model",
        model_key = bass["key"],
        payload = Dict(
            "inputs" => Dict("time" => time),
            "model_kwargs" => Dict{String,Any}(),
        ),
    )
    @test_throws Innovate.KernelBridgeError Innovate.kernel_fit_model(bad_request)
end
