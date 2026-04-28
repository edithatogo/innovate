using Innovate.Kernel;

namespace Innovate.Kernel.Tests;

public sealed class SchemaCompatibilityTests
{
    [Fact]
    public void CSharpBindingExposesSharedSchemaVersion()
    {
        Assert.Equal("1.0", KernelBinding.SchemaVersion);
    }

    [Fact]
    public void BridgeScriptLivesInCSharpBindingPackage()
    {
        Assert.EndsWith(Path.Combine("bindings", "csharp", "inst", "python", "kernel_bridge.py"), KernelBinding.BridgeScriptPath());
        Assert.True(File.Exists(KernelBinding.BridgeScriptPath()));
    }
}
