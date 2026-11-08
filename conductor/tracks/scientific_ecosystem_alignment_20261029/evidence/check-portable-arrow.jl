using Arrow, Test

direct = length(ARGS) == 3 && ARGS[1] == "--direct"
(direct || length(ARGS) == 2) ||
    error("Usage: check-portable-arrow.jl [--direct] FILE STREAM")
source, stream = read.(ARGS[end-1:end])
if !direct
    using Innovate
end

@testset "Portable Polars IPC handoff" begin
    expected = Arrow.Table(stream)
    portable = Arrow.Table(direct ? source : export_ipc_file(source))
    columns = propertynames(expected)
    @test !isempty(columns)
    @test propertynames(portable) == columns
    for name in columns
        actual = getproperty(portable, name)
        reference = getproperty(expected, name)
        @test eltype(actual) == eltype(reference)
        @test isequal(collect(actual), collect(reference))
    end
end
