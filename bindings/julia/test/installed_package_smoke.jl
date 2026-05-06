using Test

@testset "Julia installed-package bridge smoke" begin
    @test Innovate.kernel_repo_root_or_nothing() === nothing
    @test endswith(
        Innovate.kernel_bridge_script(),
        joinpath("inst", "python", "kernel_bridge.py")
    )
    @test Innovate.kernel_schema_version() == "1.0"
    @test Innovate.kernel_request(operation = "discover_models")["operation"] == "discover_models"

    if get(ENV, "INNOVATE_JULIA_RUN_BRIDGE_SMOKE", "false") == "true"
        models = Innovate.kernel_discover_models()
        @test models isa AbstractVector
        @test any(record -> record["key"] == "bass", models)
    end
end
