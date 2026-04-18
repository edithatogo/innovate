using Test
using Innovate

@testset "Julia binding packaging" begin
    example = include(joinpath(@__DIR__, "..", "examples", "end_to_end.jl"))

    @test example.schema_version == Innovate.kernel_schema_version()
    @test endswith(example.bridge_script, joinpath("inst", "python", "kernel_bridge.py"))
    @test !isempty(example.discovery)
    @test example.request["operation"] == "discover_models"
    @test example.fit["state"]["model_key"] == "bass"
    @test haskey(example.fit, "diagnostics")
    @test haskey(example.diagnose, "diagnostics")
end
