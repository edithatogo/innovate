using Test

@testset "Julia binding runtime paths" begin
    @test Innovate.kernel_repo_root() == normpath(joinpath(@__DIR__, "..", "..", ".."))
    @test Innovate._kernel_repo_root_from(mktempdir()) === nothing
    @test endswith(
        Innovate.kernel_bridge_script(),
        joinpath("inst", "python", "kernel_bridge.py")
    )
    @test Innovate.kernel_python_command() == "uv run python"
end
