# Innovate.Kernel

Thin .NET binding over the shared `innovate` functional kernel contract.

The package preserves the Python reference semantics by invoking the same JSON
kernel bridge used by the other language bindings. It does not reimplement model
behavior in C#.

## Development

```bash
dotnet test bindings/csharp/Innovate.Kernel.Tests/Innovate.Kernel.Tests.csproj
dotnet pack bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj --configuration Release
```

Set `INNOVATE_PYTHON_COMMAND` to choose the Python launcher. The default is
`uv run python` from the repository root.
