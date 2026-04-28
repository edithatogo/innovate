using Innovate.Kernel;

namespace Innovate.Kernel.Tests;

public sealed class KernelBridgeTests
{
    [Fact]
    public async Task DiscoverModelsInvokesSharedKernelBridge()
    {
        var response = await KernelBinding.DiscoverModelsAsync(TestContext.Current.CancellationToken);

        Assert.Equal(KernelBinding.SchemaVersion, response.SchemaVersion);
        Assert.Null(response.Error);
        Assert.NotNull(response.Models);
        Assert.Contains("bass", response.Models!.Value.GetRawText());
    }
}
