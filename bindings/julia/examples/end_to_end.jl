include(joinpath(@__DIR__, "..", "src", "Innovate.jl"))

using .Innovate

time = [0.0, 1.0, 2.0, 3.0, 4.0]
observed = [0.02, 0.06, 0.12, 0.25, 0.41]

function main()
    discovery = Innovate.kernel_discover_models()
    bass = first(filter(record -> record["key"] == "bass", discovery))
    fit = Innovate.kernel_fit_model(
        Innovate.kernel_request(
            operation = "fit_model",
            model_key = bass["key"],
            payload = Dict(
                "inputs" => Dict("time" => time, "observed" => observed),
                "model_kwargs" => Dict{String,Any}(),
            ),
        ),
    )
    prediction = Innovate.kernel_predict_model(
        Innovate.kernel_request(
            operation = "predict_model",
            model_key = bass["key"],
            payload = Dict(
                "inputs" => Dict("time" => time),
                "state" => fit["state"],
            ),
        ),
    )
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

    return (
        schema_version = Innovate.kernel_schema_version(),
        bridge_script = Innovate.kernel_bridge_script(),
        request = Innovate.kernel_request(operation = "discover_models"),
        discovery = discovery,
        fit = fit,
        prediction = prediction,
        summarize = summarize,
        diagnose = diagnose,
    )
end

main()
