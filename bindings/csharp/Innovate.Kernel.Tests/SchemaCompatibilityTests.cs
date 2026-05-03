using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml.Linq;
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

    [Fact]
    public void SchemaFixturesRoundTripThroughCSharpEnvelopeTypes()
    {
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web);

        var request = JsonSerializer.Deserialize<KernelRequest>(
            File.ReadAllText(FixturePath("kernel_request.predict_model.json")),
            options
        );
        var response = JsonSerializer.Deserialize<KernelResponse>(
            File.ReadAllText(FixturePath("kernel_response.predict_model.json")),
            options
        );
        var errorResponse = JsonSerializer.Deserialize<KernelResponse>(
            File.ReadAllText(FixturePath("kernel_response.error.json")),
            options
        );

        Assert.NotNull(request);
        Assert.Equal(KernelBinding.SchemaVersion, request!.SchemaVersion);
        Assert.Equal(KernelOperation.PredictModel, request.Operation);
        Assert.Equal("bass", request.ModelKey);
        Assert.Contains("t", request.Payload.Keys);

        Assert.NotNull(response);
        Assert.Equal(KernelBinding.SchemaVersion, response!.SchemaVersion);
        Assert.Equal(KernelOperation.PredictModel, response.Operation);
        Assert.Null(response.Error);
        Assert.Equal("float64", response.Result?.GetProperty("dtype").GetString());

        Assert.NotNull(errorResponse);
        Assert.Equal("invalid_payload", errorResponse!.Error?.Code);
        Assert.Equal(KernelOperation.PredictModel, errorResponse.Error?.Operation);
        Assert.False(errorResponse.Error?.Retryable);
    }

    [Fact]
    public void CSharpBindingStaysThinOverKernelEnvelopeAndBridgeTypes()
    {
        var allowedTypes = new HashSet<string>
        {
            "KernelBinding",
            "KernelBridgeException",
            "KernelError",
            "KernelOperation",
            "KernelRequest",
            "KernelResponse",
        };
        var productionFiles = Directory.GetFiles(SourceRoot(), "*.cs", SearchOption.TopDirectoryOnly);
        var typeNamePattern = new Regex(@"\b(?:class|record)\s+([A-Z][A-Za-z0-9_]*)", RegexOptions.Compiled);

        foreach (var path in productionFiles)
        {
            var source = File.ReadAllText(path);
            var declaredTypes = typeNamePattern.Matches(source).Select(match => match.Groups[1].Value);

            Assert.All(declaredTypes, typeName => Assert.Contains(typeName, allowedTypes));
            Assert.DoesNotContain("ScipyFitter", source);
            Assert.DoesNotContain("curve_fit", source);
            Assert.DoesNotContain("least_squares", source);
            Assert.DoesNotContain("BassModel", source);
            Assert.DoesNotContain("LogisticModel", source);
            Assert.DoesNotContain("GompertzModel", source);
            Assert.DoesNotContain("Math.", source);
            Assert.DoesNotContain("MathF.", source);
        }
    }

    [Fact]
    public void NuGetPackageMetadataIncludesRuntimeBridgeAsset()
    {
        var project = XDocument.Load(Path.Combine(SourceRoot(), "Innovate.Kernel.csproj"));
        var packedAssets = project
            .Descendants("None")
            .Where(element => string.Equals((string?)element.Attribute("Pack"), "true", StringComparison.OrdinalIgnoreCase))
            .Select(element => new
            {
                Include = (string?)element.Attribute("Include"),
                PackagePath = (string?)element.Attribute("PackagePath"),
                CopyToOutput = (string?)element.Attribute("PackageCopyToOutput"),
            })
            .ToList();

        Assert.Contains(packedAssets, asset => asset.Include == "../README.md" && asset.PackagePath == "/");
        Assert.Contains(
            packedAssets,
            asset =>
                asset.Include == "../inst/python/kernel_bridge.py"
                && asset.PackagePath == "contentFiles/any/any/innovate/kernel_bridge.py"
                && string.Equals(asset.CopyToOutput, "true", StringComparison.OrdinalIgnoreCase)
        );
    }

    private static string FixturePath(string fileName) =>
        Path.Combine(
            KernelBinding.RepositoryRoot(),
            "bindings",
            "csharp",
            "Innovate.Kernel.Tests",
            "fixtures",
            fileName
        );

    private static string SourceRoot() =>
        Path.Combine(KernelBinding.RepositoryRoot(), "bindings", "csharp", "Innovate.Kernel");
}
