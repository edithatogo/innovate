using System.Text.Json;
using Innovate.Kernel;

namespace Innovate.Kernel.Tests;

public sealed class KernelContractTests
{
    [Fact]
    public void KernelOperationsExposeStableContractNames()
    {
        Assert.Contains("discover_models", KernelOperation.All);
        Assert.Contains("fit_model", KernelOperation.All);
        Assert.Contains("predict_model", KernelOperation.All);
        Assert.Contains("simulate_model", KernelOperation.All);
        Assert.Contains("summarize_model", KernelOperation.All);
        Assert.Contains("diagnose_model", KernelOperation.All);
    }

    [Fact]
    public void KernelRequestSerializesUsingSharedSchemaNames()
    {
        var request = new KernelRequest
        {
            Operation = KernelOperation.DiscoverModels,
            Metadata = new Dictionary<string, object?> { ["caller"] = "csharp-test" },
        };

        var json = JsonSerializer.Serialize(request, new JsonSerializerOptions(JsonSerializerDefaults.Web));

        Assert.Contains("\"schema_version\"", json);
        Assert.Contains("\"operation\"", json);
        Assert.Contains("\"model_key\"", json);
        Assert.Contains("\"discover_models\"", json);
    }

    [Fact]
    public void KernelErrorDeserializesStableErrorEnvelope()
    {
        const string json = """
        {
          "schema_version": "1.0",
          "operation": "discover_models",
          "result": null,
          "metadata": {},
          "error": {
            "code": "invalid_request",
            "message": "bad request",
            "operation": "discover_models",
            "details": {},
            "retryable": false
          }
        }
        """;

        var response = JsonSerializer.Deserialize<KernelResponse>(json, new JsonSerializerOptions(JsonSerializerDefaults.Web));

        Assert.NotNull(response);
        Assert.Equal("invalid_request", response!.Error?.Code);
        Assert.False(response.Error?.Retryable);
    }
}
