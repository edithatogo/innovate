"""
    Innovate

Julia bindings for the Rust-owned Innovate scientific kernel.

Public helpers send schema-2.0 JSON requests or Arrow IPC through the native C ABI.
Install the matching native library in `bindings/julia/deps`, or set
`INNOVATE_NATIVE_LIBRARY` before calling a helper. Loading this module does not
build the library, execute a model or start a Python interpreter.

`kernel_call` retains complete response envelopes; convenience preprocessing and
batch helpers select documented result fields. Returned JSON nulls become
`nothing`, except `kernel_sarima_fit`, which maps warmup nulls to `missing`.
Native errors are raised rather than silently replaced by numerical fallbacks.

The docstrings describe the source contract. Runtime, scientific and distribution
qualification require the repository's Julia/native test suites and their evidence.
"""
module Innovate
using JSON
using Libdl

export invoke_registered_extension, export_ipc_file, export_ipc_stream, kernel_paired_outcomes_ipc, kernel_scenario_comparisons_ipc, kernel_scenario_executions_ipc, kernel_scenario_grid_ipc, DatasetValidationError, kernel_sarima_fit, kernel_seasonal_decomposition, kernel_aggregate_dates, kernel_cumulative_sum, kernel_rolling_average, kernel_call, kernel_discover_models, kernel_fit_model, kernel_predict_model,
    kernel_categorize_adopters, kernel_categorize_fitted_adopters, kernel_fit_batch, kernel_predict_batch, kernel_simulate_model, kernel_summarize_model, kernel_diagnose_model

"""
    DatasetValidationError(code::String, message::String, validation::Dict{String,Any})

A native request rejection carrying a structured dataset-validation report.

# Fields
- `code`: Machine-readable native error code, such as `"invalid_request"`.
- `message`: Human-readable native error message; `showerror` prints it after the code.
- `validation`: Decoded validation dictionary, retained for inspection of failed checks
  and other report fields. The constructor stores this dictionary without a deep copy.

The response decoder throws this exception only when the native error envelope
contains a `validation` member. Constructing the Julia value itself does not run
validation or establish that a dataset is scientifically suitable.
"""
struct DatasetValidationError <: Exception
    code::String
    message::String
    validation::Dict{String,Any}
end

"""
    Base.showerror(io::IO, err::DatasetValidationError)

Write the native error code and message to the supplied IO stream.

The structured validation report remains available on `err.validation`; it is not
expanded into the short exception display. Returns the result of `print`.
"""
Base.showerror(io::IO, err::DatasetValidationError) = print(io, err.code, ": ", err.message)

"""
    native_library()

Resolve the configured or bundled native shared-library path.

An explicit `INNOVATE_NATIVE_LIBRARY` overrides the platform-specific `deps`
location. Windows also recognizes Cargo's `innovate_ffi.dll` name. A missing file
raises an error; this helper neither downloads nor compiles the library.
"""
function native_library()
    bundled = joinpath(@__DIR__, "..", "deps", "libinnovate_ffi." * Libdl.dlext)
    if Sys.iswindows()
        cargo_dll = joinpath(@__DIR__, "..", "deps", "innovate_ffi.dll")
        bundled = isfile(cargo_dll) ? cargo_dll : bundled
    end
    path = get(ENV, "INNOVATE_NATIVE_LIBRARY", bundled)
    isfile(path) || error("Install the native library or set INNOVATE_NATIVE_LIBRARY")
    return abspath(path)
end

"""
    kernel_call(operation::AbstractString, model_key=nothing, payload=Dict{String,Any}())

Execute a schema-2.0 JSON request in the Rust kernel and decode its response.

The wrapper serializes and copies JSON across the C ABI; it does not call Python or supply a numerical fallback. Native response memory is released after copying its text. Loaded JSON-call handles remain cached to preserve process-local registry state. The library comes from `INNOVATE_NATIVE_LIBRARY` or the bundled `deps` directory. Integer precision after parsing follows the pinned JSON.jl decoder; use the string-accepting Arrow exporters when already holding exact JSON tokens.

# Arguments
- `operation`: Native operation name, for example `"discover_models"` or `"predict_model"`.
- `model_key`: Native model identifier, or `nothing` for an operation without a model; defaults to `nothing`.
- `payload`: JSON-compatible operation-specific dictionary; defaults to an empty `Dict{String,Any}`. Use vectors for arrays even when they contain one value, and `nothing` for JSON null.

# Returns
A `Dict{String,Any}` containing the complete successful native envelope, including `result` and any metadata. Successful responses may omit `error`; a non-null error is raised before return. JSON arrays decode as vectors and nulls as `nothing`.

# Throws
`DatasetValidationError` when a native error includes a validation report; `ErrorException` for other native errors, a missing library or a null response allocation. JSON serialization/parsing and dynamic-library loading/symbol errors propagate.

# Examples
```julia
using Innovate
response = kernel_call("discover_models")
@assert get(response, "error", nothing) === nothing
@assert any(model -> model["key"] == "bass", response["result"]["models"])
```
"""
function kernel_call(operation::AbstractString, model_key=nothing, payload=Dict{String,Any}())
    request = JSON.json(Dict("schema_version" => "2.0", "operation" => operation,
        "model_key" => model_key, "payload" => payload))
    return with_native_response(handle -> ccall(Libdl.dlsym(handle, :innovate_call),
        Ptr{UInt8}, (Cstring,), request))
end

"""
    invoke_registered_extension(callback, selection, payload)

Resolve a registered extension, run a trusted Julia callback, and validate its output.

The callback executes in the current Julia process with its normal permissions. Manifest and output validation do not sandbox it or establish the scientific validity of its computation.

# Arguments
- `callback`: Callable invoked as `callback(resolved, payload)` after native manifest resolution. Its output must satisfy the selected extension point.
- `selection`: JSON-compatible dictionary for `resolve_extension_manifest`, including `name`, `expected_version` and `extension_point`.
- `payload`: Application value passed unchanged to the callback; it is not serialized before callback invocation.

# Returns
The validated native result without its outer envelope: an ingested dataset artifact, resolved model aliases, a diagnostics summary, or an imported artifact result according to the extension point.

# Throws
Propagates `kernel_call` failures during resolution or output validation and any callback exception unchanged. Serialization fails if selection or callback output is not JSON-compatible.
"""
function invoke_registered_extension(callback, selection, payload)
    resolved = kernel_call("resolve_extension_manifest", nothing, selection)["result"]
    output = callback(resolved, payload)
    if resolved["extension_point"] == "dataset_provider"
        return kernel_call("ingest_dataset", nothing, Dict("dataset" => output))["result"]
    end
    if resolved["extension_point"] == "model_registry"
        return kernel_call("resolve_extension_model_aliases", nothing, Dict("aliases" => output))["result"]
    end
    if resolved["extension_point"] == "diagnostics"
        return kernel_call("summarize_extension_diagnostics", nothing, Dict("checks" => output))["result"]
    end
    return kernel_call("import_extension_artifact", nothing, Dict("artifact" => output))["result"]
end

"""
    kernel_scenario_grid_ipc(data::Vector{UInt8}, payload)

Execute a native scenario grid using an Arrow table as the shared time input.

Input bytes are borrowed for the synchronous C call. The native response is JSON copied into Julia and freed after decoding; no zero-copy table API or Arrow.jl dependency is provided here.

# Arguments
- `data`: A `Vector{UInt8}` containing an Arrow IPC file or stream with the required numeric `time` column.
- `payload`: JSON-compatible scenario-grid configuration, including its ordered `scenarios` and any comparisons or execution options. This is the grid payload, not a complete schema envelope.

# Returns
The decoded native response envelope. Its `result` retains ordered execution rows and any requested comparisons or artifacts. Per-scenario failures remain in their row; an outer failure is raised.

# Throws
Propagates native request, Arrow decoding, JSON serialization and library failures through the same response decoder as `kernel_call`. Other input types fail Julia method dispatch.
"""
function kernel_scenario_grid_ipc(data::Vector{UInt8}, payload)
    request = JSON.json(payload)
    return with_native_response(handle -> ccall(Libdl.dlsym(handle, :innovate_scenario_grid_ipc),
        Ptr{UInt8}, (Ptr{UInt8}, Csize_t, Cstring), data, length(data), request))
end

"""
    kernel_scenario_executions_ipc(response)

Encode ordered scenario execution rows as an Arrow IPC file.

Native-owned output bytes are copied into Julia and released before return. The wrapper does not deserialize the table into Arrow.jl or reinterpret its nested scientific results.

# Arguments
- `response`: A complete scenario-grid response as a JSON-compatible dictionary or an existing JSON string. A string is passed to Rust without re-encoding its numeric tokens.

# Returns
A new `Vector{UInt8}` containing an Arrow IPC file in native execution-table column order. Nested result/error structures are represented by the native JSON columns; row order is retained.

# Throws
An `ErrorException` for native export rejection or a null output allocation. JSON conversion, library loading and symbol errors propagate.
"""
kernel_scenario_executions_ipc(response) = scenario_output(response, :innovate_scenario_executions_ipc)

"""
    kernel_scenario_comparisons_ipc(response)

Encode ordered scenario comparison rows as an Arrow IPC file.

Native-owned output bytes are copied into Julia and released before return. The wrapper does not deserialize the table into Arrow.jl or reinterpret its nested scientific results.

# Arguments
- `response`: A complete scenario-grid response as a JSON-compatible dictionary or an existing JSON string. A string is passed to Rust without re-encoding its numeric tokens.

# Returns
A new `Vector{UInt8}` containing an Arrow IPC file in native comparison-table column order. Nested result/error structures are represented by the native JSON columns; row order is retained.

# Throws
An `ErrorException` for native export rejection or a null output allocation. JSON conversion, library loading and symbol errors propagate.
"""
kernel_scenario_comparisons_ipc(response) = scenario_output(response, :innovate_scenario_comparisons_ipc)

"""
    kernel_paired_outcomes_ipc(payload)

Validate paired Bass outcome trajectories and encode their ordered rows as Arrow IPC.

Passing a JSON string avoids a parse/re-encode cycle for full-width integer identifiers. Validation checks the interchange contract; it does not establish the calibration of the model or uncertainty draws.

# Arguments
- `payload`: Paired-outcome contract containing `strategies` and their aligned `draws`, supplied as a JSON-compatible dictionary or exact JSON string. Draws carry their identifiers and `time`/`adoption` arrays.

# Returns
A copied `Vector{UInt8}` containing an Arrow IPC file with the validated paired-outcome rows and native schema metadata.

# Throws
An `ErrorException` if trajectory shape, pairing, values or identifiers violate the native contract, or output allocation fails. Serialization and library errors propagate.
"""
kernel_paired_outcomes_ipc(payload) = scenario_output(payload, :innovate_paired_outcomes_ipc)

"""
    scenario_output(response, symbol)

Serialize a structured export input or preserve an existing JSON string.

Dispatch the named C ABI export through `arrow_output`, which copies and releases
the native byte buffer. This helper does not reinterpret full-width identifiers
in JSON strings or validate their scientific meaning.
"""
function scenario_output(response, symbol)
    request = response isa AbstractString ? response : JSON.json(response)
    return arrow_output(symbol) do convert, data, len
        ccall(convert, Ptr{UInt8},
            (Cstring, Ref{Ptr{UInt8}}, Ref{Csize_t}), request, data, len)
    end
end

"""
    export_ipc_file(input::Vector{UInt8})

Read native-supported Arrow IPC bytes and rewrite them in file format.

A file includes the footer required for random-access interchange. Rewriting is a serialization conversion, not a promise of byte identity with the input or a scientific transformation of its values.

# Arguments
- `input`: A `Vector{UInt8}` containing a valid Arrow IPC file or stream; both input encodings are accepted by the native reader.

# Returns
A new `Vector{UInt8}` containing an Arrow IPC file. Native output memory is copied into Julia and then released.

# Throws
An `ErrorException` for invalid or unsupported IPC, native conversion errors or null output allocation. Library/symbol errors propagate; non-byte-vector inputs fail method dispatch.
"""
function export_ipc_file(input::Vector{UInt8})
    return arrow_output(:innovate_export_ipc_file) do convert, data, len
        ccall(convert, Ptr{UInt8},
            (Ptr{UInt8}, Csize_t, Ref{Ptr{UInt8}}, Ref{Csize_t}), input, length(input), data, len)
    end
end

"""
    export_ipc_stream(input::Vector{UInt8})

Read native-supported Arrow IPC bytes and rewrite them in stream format.

The leading schema message carries custom schema metadata for stream interchange. Rewriting is a serialization conversion, not a promise of byte identity with the input or a scientific transformation of its values.

# Arguments
- `input`: A `Vector{UInt8}` containing a valid Arrow IPC file or stream; both input encodings are accepted by the native reader.

# Returns
A new `Vector{UInt8}` containing an Arrow IPC stream. Native output memory is copied into Julia and then released.

# Throws
An `ErrorException` for invalid or unsupported IPC, native conversion errors or null output allocation. Library/symbol errors propagate; non-byte-vector inputs fail method dispatch.
"""
function export_ipc_stream(input::Vector{UInt8})
    return arrow_output(:innovate_export_ipc_stream) do convert, data, len
        ccall(convert, Ptr{UInt8},
            (Ptr{UInt8}, Csize_t, Ref{Ptr{UInt8}}, Ref{Csize_t}), input, length(input), data, len)
    end
end

"""
    arrow_output(invoke, symbol)

Invoke one native Arrow converter and transfer its result into Julia-owned bytes.

Open the library, resolve the converter and release functions, and free native
error text or output bytes in the corresponding `finally` block. The handle is
closed after conversion. Malformed errors, missing symbols and allocation
failures propagate; the returned vector never borrows freed native memory.
"""
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

"""
    with_native_response(invoke)

Decode a copied JSON response while preserving process-local native state.

Cache library handles under a lock, invoke the supplied C-call closure, and
release its allocated response exactly once after copying. Convert native errors
with validation details into `DatasetValidationError`, and other native errors
into ordinary Julia errors. Cached library handles are intentionally retained.
"""
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

"""
    kernel_discover_models()

Return the native model catalog and its advertised capabilities.

Use the returned capabilities before choosing an operation. An advertised model does not imply that every constructor variant, simulation engine or advanced operation is available.

# Returns
The discovery `result` dictionary, including its `models` array. Model entries contain native keys and operation/capability metadata; the outer request envelope is omitted.

# Throws
Propagates `kernel_call` library, decoding and native errors. Discovery does not fit or simulate a model.
"""
kernel_discover_models() = kernel_call("discover_models")["result"]
"""
    kernel_fit_model(model, payload)

Fit one native model from explicit JSON-compatible observations and options.

Successful execution is not a guarantee of parameter identifiability, global optimality or forecast calibration. Inspect the returned optimizer and diagnostic fields for the selected model.

# Arguments
- `model`: Model key returned by `kernel_discover_models`, for example `"logistic"`.
- `payload`: Dictionary with `inputs` containing aligned `time` (or `t`) and `observed` (or `y`). Optional `constructor_kwargs` select the model variant and `fit_options` control initialization, weights and fitting. Multiseries observations use the native model-specific layout.

# Returns
The full native response envelope. Its `result` contains fitted parameters and reusable `state`, with model-specific optimizer and diagnostic details. Preserve the complete state for later prediction rather than reconstructing it from parameters.

# Throws
Propagates `kernel_call` failures for invalid observation shape, unsupported variants, parameter constraints, fitting failures or serialization/library errors.

# Examples
```julia
using Innovate
time = collect(0.0:1.0:12.0)
parameters = Dict("L" => 100.0, "k" => 0.5, "x0" => 6.0)
observed = kernel_predict_model("logistic", Dict("inputs" => Dict("time" => time),
    "parameters" => parameters))["result"]["values"]
fit = kernel_fit_model("logistic", Dict("inputs" => Dict("time" => time,
    "observed" => observed), "fit_options" => Dict("initial_parameters" => parameters)))
future = kernel_predict_model("logistic", Dict("inputs" => Dict("time" => [13.0, 14.0]),
    "state" => fit["result"]["state"]))
@assert length(future["result"]["values"]) == 2
```
"""
kernel_fit_model(model, payload) = kernel_call("fit_model", model, payload)
"""
    kernel_predict_model(model, payload)

Evaluate native level and rate predictions from explicit parameters or fitted state.

Levels and rates are different quantities. The output does not supply calibrated prediction intervals merely because a deterministic curve was evaluated.

# Arguments
- `model`: Native model identifier for the supplied parameters or state.
- `payload`: Dictionary with `inputs.time` (or `inputs.t`) and either native `parameters` with optional `constructor_kwargs`, or a retained `state`. Supply model-required covariates and options using their native field names.

# Returns
The full native response envelope. Its `result` contains model-specific prediction fields such as scalar `values`, multiseries `rows`, `rates` and reusable `state`. Arrays follow the supplied observation order; rates may have multiple columns.

# Throws
Propagates `kernel_call` failures for unsupported models, invalid grids, incompatible state/options, missing covariates or numerical-domain violations, as well as serialization/library errors.

# Examples
```julia
using Innovate
response = kernel_predict_model("logistic", Dict(
    "inputs" => Dict("time" => [1.0]),
    "parameters" => Dict("L" => 100.0, "k" => 1.0, "x0" => 1.0)))
@assert response["result"]["values"] == [50.0]
```
"""
kernel_predict_model(model, payload) = kernel_call("predict_model", model, payload)
"""
    kernel_fit_batch(model, datasets; workers=1)

Fit independent native payloads while retaining each item's outcome.

# Arguments
- `model`: Native model identifier shared by every dataset in the batch.
- `datasets`: Ordered vector of `fit_model` payload dictionaries, each carrying its own inputs and parameters/state or fitting options. These are payloads, not complete request envelopes.
- `workers`: Keyword integer from 1 through 32; defaults to 1. Parallel execution preserves dataset order.

# Returns
A vector with one dictionary per input, in the original order. Each dictionary carries `result` and `error`; an individual dataset failure remains in its row instead of being raised.

# Throws
Propagates `kernel_call` failures for an invalid outer batch, worker count or serialization/library problem. Callers must inspect per-item errors separately.
"""
kernel_fit_batch(model, datasets; workers=1) = kernel_call("fit_batch", model, Dict("datasets" => datasets, "workers" => workers))["result"]["datasets"]
"""
    kernel_predict_batch(model, datasets; workers=1)

Predict independent native payloads while retaining each item's outcome.

# Arguments
- `model`: Native model identifier shared by every dataset in the batch.
- `datasets`: Ordered vector of `predict_model` payload dictionaries, each carrying its own inputs and parameters/state or fitting options. These are payloads, not complete request envelopes.
- `workers`: Keyword integer from 1 through 32; defaults to 1. Parallel execution preserves dataset order.

# Returns
A vector with one dictionary per input, in the original order. Each dictionary carries `result` and `error`; an individual dataset failure remains in its row instead of being raised.

# Throws
Propagates `kernel_call` failures for an invalid outer batch, worker count or serialization/library problem. Callers must inspect per-item errors separately.
"""
kernel_predict_batch(model, datasets; workers=1) = kernel_call("predict_batch", model, Dict("datasets" => datasets, "workers" => workers))["result"]["datasets"]
"""
    kernel_simulate_model(model, payload)

Run a native stochastic model with an explicit engine and simulation options.

A seed supports replay under the qualified runtime and configuration. This wrapper does not promise cross-version replay or validate empirical model suitability.

# Arguments
- `model`: Native model key whose advertised capabilities support the chosen simulation engine.
- `payload`: Dictionary containing `inputs`, `parameters`, optional `constructor_kwargs`, and `simulation_options` such as `engine`, `seed`, `population`, network structure or scheduled events.

# Returns
The complete native envelope with engine-specific trajectories, state and metadata. Count-based engines may supply both integer `counts` and scaled `values`; retain the metadata when recording an experiment.

# Throws
Propagates `kernel_call` errors for unsupported engine/model combinations, invalid seeds or population/network/event settings, numerical limits and serialization/library failures.
"""
kernel_simulate_model(model, payload) = kernel_call("simulate_model", model, payload)
"""
    kernel_summarize_model(model, payload)

Request the selected model's native analytical or numerical summary.

Summary quantities inherit the supplied model assumptions and units. Availability of a summary does not establish causal or forecast validity.

# Arguments
- `model`: Native model identifier defining the supported summary operations.
- `payload`: JSON-compatible parameters or retained state plus model-specific constructor and summary inputs, such as threshold settings or a time grid when required.

# Returns
The complete response envelope with the model-specific summary in `result`. Summary fields depend on model and variant; they are not coerced into a common Julia table.

# Throws
Propagates `kernel_call` errors for unsupported summaries, missing inputs, invalid model state or numerical-domain and serialization/library failures.
"""
kernel_summarize_model(model, payload) = kernel_call("summarize_model", model, payload)
"""
    kernel_diagnose_model(model, payload)

Compute native model diagnostics from supplied observations and prediction state.

In-sample diagnostic metrics do not establish out-of-sample calibration or identify a causal effect.

# Arguments
- `model`: Native model identifier for the observations and fitted state.
- `payload`: Model-specific dictionary containing observation inputs and the parameters or fitted `state` required for prediction and diagnostic comparison.

# Returns
The full native envelope with model-specific diagnostic metrics and details in `result`. Inspect their sample and model assumptions before comparing fits.

# Throws
Propagates `kernel_call` failures for misaligned observations, unsupported diagnostics, incompatible state, numerical-domain problems or serialization/library errors.
"""
kernel_diagnose_model(model, payload) = kernel_call("diagnose_model", model, payload)
"""
    kernel_categorize_adopters(time, rates)

Assign historical adopter categories from sample-weighted observation times.

Weighted time mean and standard deviation define the five labels. Rates are treated as sample weights without multiplication by time-step widths; this does not classify individual people or integrate an irregular grid.

# Arguments
- `time`: Nonempty finite numeric vector in strictly increasing order, using one consistent time unit.
- `rates`: Aligned finite nonnegative numeric weights with positive total mass. Negative net adoption rates are not admissible weights.

# Returns
A dictionary with aligned `time`, `adoption_rate` and string `category` arrays, including a label for every zero-weight sample. Boundary ties go to the earlier historical category.

# Throws
Propagates `kernel_call` errors for empty/misaligned arrays, nonfinite values, unsorted times, negative weights or zero total mass, plus serialization/library failures.

# Examples
```julia
using Innovate
result = kernel_categorize_adopters([0., 1., 2., 3., 4.], [0., 3., 3., 1., 1.])
@assert result["category"] == ["Innovators", "Early Adopters", "Early Majority",
    "Late Majority", "Laggards"]
```
"""
kernel_categorize_adopters(time, rates) = kernel_call("categorize_adopters", nothing,
    Dict("time" => time, "adoption_rate" => rates))["result"]

"""
    kernel_categorize_fitted_adopters(model, payload)

Predict one native rate series and apply historical sample-weighted categories.

The wrapper flattens the single rate column and applies the same nonnegative sample-weight contract. Multiseries or signed substitution rates require a scientifically justified choice of admissible series outside this helper.

# Arguments
- `model`: Native model identifier whose prediction returns exactly one rate per observation time.
- `payload`: Prediction payload accepted by `kernel_predict_model`, with `inputs.time` or `inputs.t` and the required parameters or fitted state.

# Returns
The same `time`, `adoption_rate` and `category` result dictionary returned by `kernel_categorize_adopters`.

# Throws
`ArgumentError` when the native `rates` field is absent or is not a vector of one-element rows. Missing input/time keys can raise `KeyError`; native prediction and categorization failures propagate.
"""
function kernel_categorize_fitted_adopters(model, payload)
    prediction = kernel_predict_model(model, payload)["result"]
    rates = get(prediction, "rates", nothing)
    rates isa AbstractVector && all(row -> row isa AbstractVector && length(row) == 1, rates) ||
        throw(ArgumentError("adopter classification requires one native rate per time"))
    inputs = payload["inputs"]
    time = haskey(inputs, "time") ? inputs["time"] : inputs["t"]
    return kernel_categorize_adopters(time, only.(rates))
end

"""
    kernel_cumulative_sum(values)

Compute native cumulative totals in the supplied sample order.

This helper calls native preprocessing. Its missing-value semantics therefore differ from a high-level Julia missing-aware reduction.

# Arguments
- `values`: Finite numeric vector in the intended accumulation order. Values are serialized to JSON; no sorting, timestamp inference or missing-value imputation occurs.

# Returns
A vector of cumulative numeric totals with the same length as `values`; an empty vector remains empty.

# Throws
Propagates `kernel_call` failures for nonnumeric or nonfinite input and cumulative totals outside the finite numerical domain, plus serialization/library errors.

# Examples
```julia
using Innovate
@assert kernel_cumulative_sum([1.0, 2.0, 3.0]) == [1.0, 3.0, 6.0]
```
"""
kernel_cumulative_sum(values) = kernel_call("preprocess_series", nothing,
    Dict("method" => "cumulative_sum", "values" => values))["result"]["values"]
"""
    kernel_rolling_average(values, window)

Compute native trailing averages only after a complete observation window exists.

The native implementation uses guarded accumulation for finite inputs. No missing-value imputation or calendar-based windowing is performed.

# Arguments
- `values`: Finite numeric vector in sample order, without missing or null entries.
- `window`: Positive integer number of consecutive samples per trailing mean; values larger than the series length leave all entries unfilled.

# Returns
A vector aligned to `values`, with `nothing` for the first `window - 1` positions (or all positions if the window is longer) and numeric complete-window means afterward.

# Throws
Propagates `kernel_call` failures for a nonpositive/noninteger window, invalid samples or nonfinite arithmetic, plus serialization/library errors.

# Examples
```julia
using Innovate
@assert kernel_rolling_average([1.0, 2.0, 3.0], 2) == [nothing, 1.5, 2.5]
```
"""
kernel_rolling_average(values, window) = kernel_call("preprocess_series", nothing,
    Dict("method" => "rolling_average", "values" => values, "window" => window))["result"]["values"]

"""
    kernel_aggregate_dates(dates, values, frequency)

Sum aligned observations into native calendar or anchored weekly buckets.

Offset timestamps normalize to UTC. Weekly timestamp windows span seven days from the earliest timestamp; nonweekly timestamp aggregation uses the normalized calendar date. Use `kernel_call("aggregate_dates", nothing, payload)` for fiscal-start controls not exposed here.

# Arguments
- `dates`: Vector converted elementwise with `string`. Supply ISO dates or consistently naive/offset ISO timestamps; do not mix dates with timestamps or offset with naive timestamps.
- `values`: Aligned finite numeric vector to sum; input order need not already be chronological.
- `frequency`: One of `"D"`, `"W"`, `"M"`, `"ME"`, `"MS"`, `"QS"`, `"QE"`, `"YS"` or `"YE"`. Quarter/year buckets use January boundaries in this wrapper.

# Returns
A dictionary containing sorted `dates` labels and aggregated `values`. Weekly buckets start at the earliest input and omit empty weeks; other supported frequencies fill intervening empty periods with zero. Month/quarter/year end variants use end labels.

# Throws
Propagates `kernel_call` errors for misalignment, invalid/mixed timestamps, unsupported frequency, nonfinite sums, date-range limits or more than 100000 output periods, plus serialization/library errors.

# Examples
```julia
using Innovate
result = kernel_aggregate_dates(["2026-11-01", "2026-11-02"], [1., 2.], "W")
@assert result["dates"] == ["2026-11-01"]
@assert result["values"] == [3.]
```
"""
kernel_aggregate_dates(dates, values, frequency) = kernel_call("aggregate_dates", nothing,
    Dict("dates" => string.(dates), "values" => values, "frequency" => frequency))["result"]

"""
    kernel_seasonal_decomposition(values, period, trend_span=nothing; robust=false, historical=false,
        inner_iterations=5, outer_iterations=robust ? 5 : 0, seasonal_span=nothing, low_pass_span=nothing,
        seasonal_degree=1, seasonal_jump=1, trend_degree=1, trend_jump=1, low_pass_degree=1, low_pass_jump=1)

Decompose an evenly spaced native series into trend, seasonal and residual components.

# Arguments
- `values`: Finite numeric vector without missing entries, ordered on an evenly spaced grid; the caller must establish spacing because this wrapper receives no timestamps.
- `period`: Integer samples per seasonal cycle, at least 2, leaving at least three complete cycles in the input.
- `trend_span`: Trend smoothing window, default `nothing`. A numeric window is required unless `historical=true`, which derives an odd span from period and seasonal span. Native span bounds apply.
- `robust`: Keyword boolean selecting residual reweighting; defaults to `false`.
- `historical`: Keyword boolean selecting historical STL local-fitting and reweighting rules; defaults to `false`.
- `inner_iterations`: Keyword integer number of inner smoothing passes, from 1 through 100; defaults to 5.
- `outer_iterations`: Keyword integer reweighting passes, from 0 through 100; defaults to 5 when robust and 0 otherwise. Positive counts require `robust=true`.
- `seasonal_span`: Optional keyword odd seasonal span of at least 3; default `nothing` selects the native default, 7 in historical mode or at most 7 phase samples otherwise.
- `low_pass_span`: Optional keyword odd low-pass span of at least 3; default `nothing` selects the smallest odd integer strictly above the period.
- `seasonal_degree`: Keyword seasonal local polynomial degree, 0 or 1; defaults to 1.
- `seasonal_jump`: Keyword positive spacing between seasonal fitted points; defaults to 1.
- `trend_degree`: Keyword trend local polynomial degree, 0 or 1; defaults to 1.
- `trend_jump`: Keyword positive spacing between trend fitted points; defaults to 1.
- `low_pass_degree`: Keyword low-pass local polynomial degree, 0 or 1; defaults to 1.
- `low_pass_jump`: Keyword positive spacing between low-pass fitted points; defaults to 1.

# Returns
A dictionary containing `trend`, `seasonal` and `residual` numeric arrays, each aligned to the input samples. This is a decomposition, not a future forecast or uncertainty estimate.

# Throws
Propagates `kernel_call` errors for too few cycles, invalid spans/iteration/smoother settings, invalid samples or nonfinite numerical estimates, plus serialization/library errors.
"""
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

"""
    kernel_sarima_fit(values, order; seasonal_order=(0, 0, 0, 0), historical=false)

Fit a no-intercept native SARIMA model and return only its in-sample fitted values.

Outside historical mode, initial missing values reflect differencing warmup. Successful fitting does not establish the adequacy of the time-series specification.

# Arguments
- `values`: Finite numeric vector without missing entries, already ordered and evenly spaced; the wrapper does not receive or validate timestamps.
- `order`: Iterable ordinary `(p, d, q)` orders, converted with `collect`; all orders must be nonnegative integers within native order/sample-size bounds.
- `seasonal_order`: Keyword iterable `(P, D, Q, period)` orders; defaults to `(0, 0, 0, 0)`. Seasonal differencing requires a period of at least 2.
- `historical`: Keyword boolean enabling historical diffuse-initialization predictions where supported; defaults to `false`.

# Returns
A vector aligned to the input, with numeric fitted entries converted to `Float64` and native null warmup entries converted to `missing`. Coefficients, future forecasts and prediction intervals are not returned.

# Throws
Propagates `kernel_call` errors for invalid orders, insufficient samples, stationarity/estimation failures or nonfinite input, plus iterable conversion, numeric conversion and serialization/library errors.
"""
kernel_sarima_fit(values, order; seasonal_order=(0, 0, 0, 0), historical=false) =
    map(x -> isnothing(x) ? missing : Float64(x),
        kernel_call("preprocess_series", nothing, Dict("method" => "sarima_fit",
            "values" => values, "order" => collect(order),
            "seasonal_order" => collect(seasonal_order), "historical" => historical))["result"]["fitted"])

end
