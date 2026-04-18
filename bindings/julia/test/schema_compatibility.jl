using Test

function python_kernel_schema_version()
    source = read(joinpath(@__DIR__, "..", "..", "..", "src", "innovate", "kernel.py"), String)
    major = match(r"KERNEL_SCHEMA_MAJOR_VERSION\s*=\s*(\d+)", source)
    minor = match(r"KERNEL_SCHEMA_MINOR_VERSION\s*=\s*(\d+)", source)

    @test major !== nothing
    @test minor !== nothing

    return string(major.captures[1], ".", minor.captures[1])
end

@testset "Julia binding schema compatibility" begin
    python_schema_version = python_kernel_schema_version()

    @test Innovate.kernel_schema_version() == python_schema_version
    @test Innovate.kernel_request(operation = "discover_models")["schema_version"] == python_schema_version
    @test isfile(Innovate.kernel_bridge_script())
    @test endswith(Innovate.kernel_bridge_script(), joinpath("inst", "python", "kernel_bridge.py"))
end
