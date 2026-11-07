module Innovate
using JSON
using Libdl

export kernel_call, kernel_discover_models, kernel_fit_model, kernel_predict_model,
    kernel_fit_batch, kernel_predict_batch, kernel_simulate_model, kernel_summarize_model, kernel_diagnose_model

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
end
