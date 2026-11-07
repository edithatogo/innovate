using Arrow, Innovate, Test

length(ARGS) == 2 || error("Supply a Polars IPC file and an equivalent IPC stream")
source, stream = read.(ARGS)

@testset "Portable Polars IPC handoff" begin
    expected = Arrow.Table(stream)
    portable = Arrow.Table(export_ipc_file(source))
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
