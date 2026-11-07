module Innovate
using JSON
using Libdl

export invoke_registered_extension, export_ipc_file, kernel_paired_outcomes_ipc, kernel_scenario_comparisons_ipc, kernel_scenario_executions_ipc, kernel_scenario_grid_ipc, DatasetValidationError, kernel_sarima_fit, kernel_seasonal_decomposition, kernel_aggregate_dates, kernel_cumulative_sum, kernel_rolling_average, kernel_call, kernel_discover_models, kernel_fit_model, kernel_predict_model,
    kernel_categorize_adopters, kernel_categorize_fitted_adopters, kernel_fit_batch, kernel_predict_batch, kernel_simulate_model, kernel_summarize_model, kernel_diagnose_model

"""A rejected dataset with its native validation report."""
struct DatasetValidationError <: Exception
    code::String
    message::String
    validation::Dict{String,Any}
end

Base.showerror(io::IO, err::DatasetValidationError) = print(io, err.code, ": ", err.message)

function native_library()
    path = get(ENV, "INNOVATE_NATIVE_LIBRARY", joinpath(@__DIR__, "..", "deps", "libinnovate_ffi." * Libdl.dlext))
    isfile(path) || error("Install the native library or set INNOVATE_NATIVE_LIBRARY")
    return abspath(path)
end

"""Execute in Rust; returned native memory is released exactly once."""
function kernel_call(operation::AbstractString, model_key=nothing, payload=Dict{String,Any}())
    request = JSON.json(Dict("schema_version" => "2.0", "operation" => operation,
        "model_key" => model_key, "payload" => payload))
    return with_native_response(handle -> ccall(Libdl.dlsym(handle, :innovate_call),
        Ptr{UInt8}, (Cstring,), request))
end

"""Resolve a registry snapshot and invoke a trusted caller-supplied callback."""
function invoke_registered_extension(callback, selection, payload)
    resolved = kernel_call("resolve_extension_manifest", nothing, selection)["result"]
    output = callback(resolved, payload)
    if resolved["extension_point"] == "dataset_provider"
        return kernel_call("ingest_dataset", nothing, Dict("dataset" => output))["result"]
    end
    return output
end

"""Execute a scenario grid on a shared time column in Arrow IPC file or stream bytes."""
function kernel_scenario_grid_ipc(data::Vector{UInt8}, payload)
    request = JSON.json(payload)
    return with_native_response(handle -> ccall(Libdl.dlsym(handle, :innovate_scenario_grid_ipc),
        Ptr{UInt8}, (Ptr{UInt8}, Csize_t, Cstring), data, length(data), request))
end

"""Encode ordered scenario execution rows as Arrow IPC file bytes."""
kernel_scenario_executions_ipc(response) = scenario_output(response, :innovate_scenario_executions_ipc)

"""Encode ordered scenario comparison rows as Arrow IPC file bytes."""
kernel_scenario_comparisons_ipc(response) = scenario_output(response, :innovate_scenario_comparisons_ipc)

"""Validate paired Bass trajectories and return Arrow IPC file bytes."""
kernel_paired_outcomes_ipc(payload) = scenario_output(payload, :innovate_paired_outcomes_ipc)

function scenario_output(response, symbol)
    request = response isa AbstractString ? response : JSON.json(response)
    return arrow_output(symbol) do convert, data, len
        ccall(convert, Ptr{UInt8},
            (Cstring, Ref{Ptr{UInt8}}, Ref{Csize_t}), request, data, len)
    end
end

"""Rewrite Arrow IPC file or stream bytes for portable file interchange."""
function export_ipc_file(input::Vector{UInt8})
    return arrow_output(:innovate_export_ipc_file) do convert, data, len
        ccall(convert, Ptr{UInt8},
            (Ptr{UInt8}, Csize_t, Ref{Ptr{UInt8}}, Ref{Csize_t}), input, length(input), data, len)
    end
end

function arrow_output(invoke, symbol)
    handle = Libdl.dlopen(native_library())
    try
        convert = Libdl.dlsym(handle, symbol)
        free_bytes = Libdl.dlsym(handle, :innovate_free_bytes)
        free_text = Libdl.dlsym(handle, :innovate_free)
        data = Ref{Ptr{UInt8}}(C_NULL)
        len = Ref{Csize_t}(0)
        failure = invoke(convert, data, len)
        if failure != C_NULL
            text = try
                unsafe_string(failure)
            finally
                ccall(free_text, Cvoid, (Ptr{UInt8},), failure)
            end
            details = JSON.parse(text)["error"]
            error(string(details["code"], ": ", details["message"]))
        end
        data[] == C_NULL && error("Native Arrow response allocation failed")
        try
            return copy(unsafe_wrap(Vector{UInt8}, data[], Int(len[])))
        finally
            ccall(free_bytes, Cvoid, (Ptr{UInt8}, Csize_t), data[], len[])
        end
    finally
        Libdl.dlclose(handle)
    end
end

const native_handles = Dict{String,Ptr{Cvoid}}()
const native_handles_lock = ReentrantLock()

function with_native_response(invoke)
    path = native_library()
    # Keep process-local Rust state alive across JSON calls.
    handle = lock(native_handles_lock) do
        get!(() -> Libdl.dlopen(path), native_handles, path)
    end
    release = Libdl.dlsym(handle, :innovate_free)
    response = invoke(handle)
    response == C_NULL && error("Native response allocation failed")
    text = try
        unsafe_string(response)
    finally
        ccall(release, Cvoid, (Ptr{UInt8},), response)
    end
    result = JSON.parse(text; dicttype=Dict{String,Any})
    if get(result, "error", nothing) !== nothing
        details = result["error"]
        if haskey(details, "validation")
            throw(DatasetValidationError(details["code"], details["message"], details["validation"]))
        end
        error(string(details["code"], ": ", details["message"]))
    end
    return result
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

kernel_aggregate_dates(dates, values, frequency) = kernel_call("aggregate_dates", nothing,
    Dict("dates" => string.(dates), "values" => values, "frequency" => frequency))["result"]

kernel_seasonal_decomposition(values, period, trend_span=nothing; robust=false, historical=false,
    inner_iterations=5, outer_iterations=robust ? 5 : 0, seasonal_span=nothing, low_pass_span=nothing,
    seasonal_degree=1, seasonal_jump=1, trend_degree=1, trend_jump=1, low_pass_degree=1, low_pass_jump=1) = kernel_call("preprocess_series", nothing,
    Dict("method" => "seasonal_decomposition", "values" => values, "period" => period,
        "window" => trend_span, "robust" => robust, "historical" => historical,
        "inner_iterations" => inner_iterations, "outer_iterations" => outer_iterations,
        "seasonal_span" => seasonal_span, "low_pass_span" => low_pass_span,
        "seasonal_degree" => seasonal_degree, "seasonal_jump" => seasonal_jump,
        "trend_degree" => trend_degree, "trend_jump" => trend_jump,
        "low_pass_degree" => low_pass_degree, "low_pass_jump" => low_pass_jump))["result"]

"""Estimate a no-intercept SARIMA model and return fitted values with missing warmup rows."""
kernel_sarima_fit(values, order; seasonal_order=(0, 0, 0, 0), historical=false) =
    map(x -> isnothing(x) ? missing : Float64(x),
        kernel_call("preprocess_series", nothing, Dict("method" => "sarima_fit",
            "values" => values, "order" => collect(order),
            "seasonal_order" => collect(seasonal_order), "historical" => historical))["result"]["fitted"])

end
