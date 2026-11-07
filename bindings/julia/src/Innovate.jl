module Innovate
using JSON
using Libdl

export kernel_cumulative_sum, kernel_rolling_average, kernel_call, kernel_discover_models, kernel_fit_model, kernel_predict_model,
    kernel_categorize_adopters, kernel_categorize_fitted_adopters, kernel_fit_batch, kernel_predict_batch, kernel_simulate_model, kernel_summarize_model, kernel_diagnose_model

function native_library()
    path = get(ENV, "INNOVATE_NATIVE_LIBRARY", joinpath(@__DIR__, "..", "deps", "libinnovate_ffi." * Libdl.dlext))
    isfile(path) || error("Install the native library or set INNOVATE_NATIVE_LIBRARY")
    return abspath(path)
end

"""Execute in Rust; returned native memory is released exactly once."""
function kernel_call(operation::AbstractString, model_key=nothing, payload=Dict{String,Any}())
    request = JSON.json(Dict("schema_version" => "2.0", "operation" => operation,
        "model_key" => model_key, "payload" => payload))
    handle = Libdl.dlopen(native_library())
    try
        call = Libdl.dlsym(handle, :innovate_call)
        release = Libdl.dlsym(handle, :innovate_free)
        response = ccall(call, Ptr{UInt8}, (Cstring,), request)
        response == C_NULL && error("Native response allocation failed")
        text = try
            unsafe_string(response)
        finally
            ccall(release, Cvoid, (Ptr{UInt8},), response)
        end
        result = JSON.parse(text; dicttype=Dict{String,Any})
        if get(result, "error", nothing) !== nothing
            error(string(result["error"]["code"], ": ", result["error"]["message"]))
        end
        return result
    finally
        Libdl.dlclose(handle)
    end
end

kernel_discover_models() = kernel_call("discover_models")["result"]
kernel_fit_model(model, payload) = kernel_call("fit_model", model, payload)
kernel_predict_model(model, payload) = kernel_call("predict_model", model, payload)
kernel_fit_batch(model, datasets; workers=1) = kernel_call("fit_batch", model, Dict("datasets" => datasets, "workers" => workers))["result"]["datasets"]
kernel_predict_batch(model, datasets; workers=1) = kernel_call("predict_batch", model, Dict("datasets" => datasets, "workers" => workers))["result"]["datasets"]
kernel_simulate_model(model, payload) = kernel_call("simulate_model", model, payload)
kernel_summarize_model(model, payload) = kernel_call("summarize_model", model, payload)
kernel_diagnose_model(model, payload) = kernel_call("diagnose_model", model, payload)
kernel_categorize_adopters(time, rates) = kernel_call("categorize_adopters", nothing,
    Dict("time" => time, "adoption_rate" => rates))["result"]

"""Classify one native rate series using a fitted-state prediction payload."""
function kernel_categorize_fitted_adopters(model, payload)
    prediction = kernel_predict_model(model, payload)["result"]
    rates = get(prediction, "rates", nothing)
    rates isa AbstractVector && all(row -> row isa AbstractVector && length(row) == 1, rates) ||
        throw(ArgumentError("adopter classification requires one native rate per time"))
    inputs = payload["inputs"]
    time = haskey(inputs, "time") ? inputs["time"] : inputs["t"]
    return kernel_categorize_adopters(time, only.(rates))
end

kernel_cumulative_sum(values) = kernel_call("preprocess_series", nothing,
    Dict("method" => "cumulative_sum", "values" => values))["result"]["values"]
kernel_rolling_average(values, window) = kernel_call("preprocess_series", nothing,
    Dict("method" => "rolling_average", "values" => values, "window" => window))["result"]["values"]

end
