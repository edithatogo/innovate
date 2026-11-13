using JSON
using Innovate

base = ENV["SCENARIO_BINDING_INPUTS"]
expected = JSON.parsefile(joinpath(base, "expected.json"))
@assert length(expected) == 24
for (index, reference) in enumerate(expected)
    prefix = lpad(index - 1, 2, '0')
    import_request = JSON.parsefile(joinpath(base, prefix * "-import.json"))
    imported = kernel_call(import_request["operation"], nothing, import_request["payload"])["result"]
    @assert imported["replay_available"]
    replay_request = JSON.parsefile(joinpath(base, prefix * "-replay.json"))
    row = kernel_call(replay_request["operation"], nothing, replay_request["payload"])["result"]["executions"][1]
    @assert isnothing(row["error"])
    for (key, value) in reference["expected"]
        @assert row["result"][key] == value
    end
end
for index in 0:4
    request = JSON.parsefile(joinpath(base, "legacy-" * lpad(index, 2, '0') * ".json"))
    result = kernel_call(request["operation"], nothing, request["payload"])["result"]
    @assert !result["replay_available"]
    @assert result["artifact"]["seed"] == typemax(UInt64)
end
println("Julia ", VERSION, " matched 24 historical replays and preserved 5 legacy seeds.")
