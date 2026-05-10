# innovate.cs

Thin .NET binding over the shared `innovate` functional kernel contract. Promoted native slices
execute in the shared contract, unsupported promoted payloads return explicit native errors, and
bridge fallback remains available only for explicitly non-native model families.

The package preserves the Python reference semantics by invoking the same JSON
kernel bridge used by the other language bindings for explicitly non-native
model families. It does not reimplement model behavior in C#.

The package targets both `net10.0` and `net11.0`.

NuGet packages include Apache-2.0 license metadata, repository metadata, SourceLink
settings, a symbol package, this readme, and the packaged Python bridge script
under `contentFiles/any/any/innovate/kernel_bridge.py`.

## Development

```bash
dotnet test bindings/csharp/Innovate.Kernel.Tests/Innovate.Kernel.Tests.csproj
dotnet pack bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj --configuration Release
```

Set `INNOVATE_PYTHON_COMMAND` to choose the Python launcher. The default is
`uv run python` from the repository root.
