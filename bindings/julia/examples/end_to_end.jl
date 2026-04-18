include(joinpath(@__DIR__, "..", "src", "Innovate.jl"))

using .Innovate

function main()
    return (
        schema_version = Innovate.kernel_schema_version(),
        bridge_script = Innovate.kernel_bridge_script(),
        request = Innovate.kernel_request(operation = "discover_models"),
    )
end

main()
