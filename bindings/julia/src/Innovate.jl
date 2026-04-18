module Innovate

export kernel_repo_root,
    kernel_bindings_root,
    kernel_python_command,
    kernel_bridge_script,
    kernel_schema_version,
    kernel_request,
    kernel_call,
    kernel_response_to_julia,
    kernel_discover_models,
    kernel_fit_model,
    kernel_predict_model,
    kernel_simulate_model,
    kernel_summarize_model,
    kernel_diagnose_model,
    kernel_extract_diagnostics

const _KERNEL_SCHEMA_VERSION = "0.1.0"

"""Return the repository root that contains the shared kernel and bindings."""
kernel_repo_root() = normpath(joinpath(@__DIR__, "..", "..", ".."))

"""Return the Julia bindings package root."""
kernel_bindings_root() = normpath(joinpath(@__DIR__, "..", ".."))

"""Return the Python launcher used by the Julia bindings."""
kernel_python_command() = get(ENV, "INNOVATE_PYTHON_COMMAND", "uv run python")

"""Return the path to the Python kernel bridge entrypoint."""
kernel_bridge_script() = joinpath(kernel_bindings_root(), "inst", "python", "kernel_bridge.py")

"""Return the schema version exposed by the kernel contract."""
kernel_schema_version() = _KERNEL_SCHEMA_VERSION

"""Create a kernel request envelope."""
function kernel_request(; operation::AbstractString, model_key::AbstractString = "", payload = Dict{String,Any}())
    return Dict{String,Any}(
        "operation" => String(operation),
        "model_key" => String(model_key),
        "payload" => payload,
    )
end

"""Placeholder kernel invocation hook for the Julia scaffold."""
function kernel_call(request::AbstractDict)
    throw(ArgumentError("Julia kernel bridge is scaffolded but not yet wired to the Python runtime"))
end

"""Normalize a response value into a Julia-friendly structure."""
kernel_response_to_julia(value) = value

"""Discover stable kernel models."""
kernel_discover_models() = kernel_call(kernel_request(operation = "discover_models"))

"""Fit a kernel model."""
kernel_fit_model(request::AbstractDict) = kernel_call(request)

"""Predict from a kernel model."""
kernel_predict_model(request::AbstractDict) = kernel_call(request)

"""Simulate a kernel model."""
kernel_simulate_model(request::AbstractDict) = kernel_call(request)

"""Summarize a kernel model."""
kernel_summarize_model(request::AbstractDict) = kernel_call(request)

"""Diagnose a kernel model."""
kernel_diagnose_model(request::AbstractDict) = kernel_call(request)

"""Extract diagnostics from a result envelope."""
function kernel_extract_diagnostics(result)
    if result isa AbstractDict && haskey(result, "diagnostics")
        return result["diagnostics"]
    end
    return nothing
end

end
