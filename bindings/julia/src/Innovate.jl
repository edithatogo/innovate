module Innovate

using JSON

export kernel_repo_root,
    kernel_bindings_root,
    kernel_python_command,
    kernel_bridge_script,
    kernel_schema_version,
    kernel_request,
    kernel_call,
    kernel_response_to_julia,
    KernelBridgeError,
    kernel_discover_models,
    kernel_fit_model,
    kernel_predict_model,
    kernel_simulate_model,
    kernel_summarize_model,
    kernel_diagnose_model,
    kernel_extract_diagnostics

const _KERNEL_SCHEMA_VERSION = "1.0"

"""Stable error payload returned by the Julia bridge."""
struct KernelBridgeError <: Exception
    code::String
    message::String
    operation::String
    details::Dict{String,Any}
    retryable::Bool
    response::Dict{String,Any}
end

Base.showerror(io::IO, error::KernelBridgeError) = print(
    io,
    "Kernel bridge error ($(error.code)) for operation '$(error.operation)': $(error.message)",
)

"""Return the repository root that contains the shared kernel and bindings."""
kernel_repo_root() = normpath(joinpath(@__DIR__, "..", "..", ".."))

"""Return the Julia bindings package root."""
kernel_bindings_root() = normpath(joinpath(@__DIR__, ".."))

"""Return the Python launcher used by the Julia bindings."""
kernel_python_command() = get(ENV, "INNOVATE_PYTHON_COMMAND", "uv run python")

"""Return the path to the Python kernel bridge entrypoint."""
kernel_bridge_script() = joinpath(kernel_bindings_root(), "inst", "python", "kernel_bridge.py")

"""Return the schema version exposed by the kernel contract."""
kernel_schema_version() = _KERNEL_SCHEMA_VERSION

function _jsonify(value)
    if value === nothing || value isa Bool || value isa Number || value isa AbstractString
        return value
    elseif value isa AbstractDict
        return Dict{String,Any}(String(key) => _jsonify(item) for (key, item) in value)
    elseif value isa NamedTuple
        return Dict{String,Any}(String(key) => _jsonify(item) for (key, item) in pairs(value))
    elseif value isa AbstractVector
        return [_jsonify(item) for item in value]
    elseif value isa Tuple
        return [_jsonify(item) for item in value]
    end

    return value
end

"""Create a kernel request envelope."""
function kernel_request(;
    operation::AbstractString,
    model_key::AbstractString = "",
    payload = Dict{String,Any}(),
    metadata = Dict{String,Any}(),
    schema_version::AbstractString = kernel_schema_version(),
)
    operation_text = String(operation)
    model_key_text = String(model_key)
    if isempty(operation_text)
        throw(ArgumentError("kernel_request() requires a non-empty operation"))
    end
    if operation_text != "discover_models" && isempty(model_key_text)
        throw(ArgumentError("Kernel operation '$operation_text' requires a model_key"))
    end

    return Dict{String,Any}(
        "schema_version" => String(schema_version),
        "operation" => operation_text,
        "model_key" => isempty(model_key_text) ? nothing : model_key_text,
        "payload" => _jsonify(payload),
        "metadata" => _jsonify(metadata),
    )
end

function _kernel_bridge_command()
    command = Base.shell_split(kernel_python_command())
    isempty(command) && throw(ArgumentError("INNOVATE_PYTHON_COMMAND must not be empty"))
    return command
end

function _kernel_bridge_raw_command(request_path::AbstractString, response_path::AbstractString)
    command = _kernel_bridge_command()
    return Cmd(vcat(command, [kernel_bridge_script(), request_path, response_path]))
end

"""Invoke the Python kernel bridge and return the raw JSON response."""
function kernel_call(request::AbstractDict)
    request_path = tempname() * ".json"
    response_path = tempname() * ".json"

    try
        open(request_path, "w") do io
            write(io, JSON.json(_jsonify(request)))
        end

        withenv("PYTHONPATH" => joinpath(kernel_repo_root(), "src")) do
            cd(kernel_repo_root()) do
                run(_kernel_bridge_raw_command(request_path, response_path))
            end
        end

        return JSON.parsefile(response_path)
    finally
        if isfile(request_path)
            rm(request_path; force = true)
        end
        if isfile(response_path)
            rm(response_path; force = true)
        end
    end
end

function _json_value_to_julia(value)
    if value === nothing || value isa Bool || value isa Number || value isa AbstractString
        return value
    elseif value isa AbstractDict
        return Dict{String,Any}(String(key) => _json_value_to_julia(item) for (key, item) in value)
    elseif value isa AbstractVector
        return [_json_value_to_julia(item) for item in value]
    end

    return value
end

function _decode_array_payload(value::AbstractDict)
    shape = Tuple(Int.(value["shape"]))
    values = Float64.(_json_value_to_julia.(value["values"]))
    if isempty(shape)
        return values
    end
    return reshape(values, shape)
end

function _decode_table_payload(value::AbstractDict)
    columns = String.(value["columns"])
    rows = value["rows"]
    names = Tuple(Symbol.(columns))
    return [
        NamedTuple{names}(Tuple(_json_value_to_julia.(row)))
        for row in rows
    ]
end

function _decode_discovery_response(value::AbstractDict)
    return [_json_value_to_julia(record) for record in value["models"]]
end

"""Normalize a response value into a Julia-friendly structure."""
function kernel_response_to_julia(value)
    if value === nothing
        return nothing
    elseif value isa AbstractDict
        if haskey(value, "error") && !isnothing(value["error"])
            error = value["error"]
            operation_value = get(error, "operation", nothing)
            operation = operation_value === nothing ? String(get(value, "operation", "discover_models")) : String(operation_value)
            details_source = get(error, "details", Dict{String,Any}())
            response = Dict{String,Any}(String(key) => _json_value_to_julia(item) for (key, item) in value)
            throw(
                KernelBridgeError(
                    String(error["code"]),
                    String(error["message"]),
                    operation,
                    details_source isa AbstractDict ? Dict{String,Any}(String(key) => _json_value_to_julia(item) for (key, item) in details_source) : Dict{String,Any}(),
                    Bool(get(error, "retryable", false)),
                    response,
                ),
            )
        end

        if haskey(value, "shape") && haskey(value, "dtype") && haskey(value, "values")
            return _decode_array_payload(value)
        elseif haskey(value, "columns") && haskey(value, "rows")
            return _decode_table_payload(value)
        elseif haskey(value, "models") && haskey(value, "schema_version") && !haskey(value, "operation")
            return _decode_discovery_response(value)
        elseif haskey(value, "operation") && haskey(value, "result") && !isnothing(value["result"])
            return kernel_response_to_julia(value["result"])
        end

        return Dict{String,Any}(String(key) => _json_value_to_julia(item) for (key, item) in value)
    elseif value isa AbstractVector
        return [_json_value_to_julia(item) for item in value]
    end

    return value
end

"""Discover stable kernel models."""
kernel_discover_models() = kernel_response_to_julia(kernel_call(kernel_request(operation = "discover_models")))

"""Fit a kernel model."""
kernel_fit_model(request::AbstractDict) = kernel_response_to_julia(kernel_call(request))

"""Predict from a kernel model."""
kernel_predict_model(request::AbstractDict) = kernel_response_to_julia(kernel_call(request))

"""Simulate a kernel model."""
kernel_simulate_model(request::AbstractDict) = kernel_response_to_julia(kernel_call(request))

"""Summarize a kernel model."""
kernel_summarize_model(request::AbstractDict) = kernel_response_to_julia(kernel_call(request))

"""Diagnose a kernel model."""
kernel_diagnose_model(request::AbstractDict) = kernel_response_to_julia(kernel_call(request))

"""Extract diagnostics from a result envelope."""
function kernel_extract_diagnostics(result)
    if result isa AbstractDict && haskey(result, "diagnostics")
        return result["diagnostics"]
    elseif result isa NamedTuple && hasproperty(result, :diagnostics)
        return getproperty(result, :diagnostics)
    end
    return nothing
end

end
