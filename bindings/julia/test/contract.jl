using Test
using Innovate

@testset "Julia binding contract" begin
    @test Innovate.kernel_schema_version() == "0.1.0"
    @test endswith(
        Innovate.kernel_bridge_script(),
        joinpath("inst", "python", "kernel_bridge.py")
    )
    @test Innovate.kernel_python_command() == "uv run python"

    request = Innovate.kernel_request(operation = "discover_models")
    @test request["operation"] == "discover_models"
    @test request["payload"] isa Dict
end
