using Test

function python_kernel_schema_version()
    source_path = joinpath(@__DIR__, "..", "..", "..", "src", "innovate", "kernel.py")
    if !isfile(source_path) || get(ENV, "INNOVATE_JULIA_RUN_INSTALLED_PACKAGE_SMOKE", "false") == "true"
        return nothing
    end

    source = read(source_path, String)
    major = match(r"KERNEL_SCHEMA_MAJOR_VERSION\s*=\s*(\d+)", source)
    minor = match(r"KERNEL_SCHEMA_MINOR_VERSION\s*=\s*(\d+)", source)

    @test major !== nothing
    @test minor !== nothing

    return string(major.captures[1], ".", minor.captures[1])
end

@testset "Julia binding schema compatibility" begin
    python_schema_version = python_kernel_schema_version()

    @test Innovate.kernel_schema_version() == "1.0"
    @test isfile(Innovate.kernel_bridge_script())
    @test endswith(Innovate.kernel_bridge_script(), joinpath("inst", "python", "kernel_bridge.py"))

    if python_schema_version === nothing
        @test Innovate.kernel_request(operation = "discover_models")["schema_version"] == "1.0"
    else
        @test Innovate.kernel_schema_version() == python_schema_version
        @test Innovate.kernel_request(operation = "discover_models")["schema_version"] == python_schema_version
    end
end
