using System.Text.Json.Serialization;

namespace Innovate.Kernel;

/// <summary>Language-neutral kernel request envelope.</summary>
public sealed record KernelRequest
{
    [JsonPropertyName("schema_version")]
    public string SchemaVersion { get; init; } = KernelBinding.SchemaVersion;

    [JsonPropertyName("operation")]
    public required string Operation { get; init; }

    [JsonPropertyName("model_key")]
    public string? ModelKey { get; init; }

    [JsonPropertyName("payload")]
    public Dictionary<string, object?> Payload { get; init; } = [];

    [JsonPropertyName("metadata")]
    public Dictionary<string, object?> Metadata { get; init; } = [];

    public static KernelRequest DiscoverModels() =>
        new() { Operation = KernelOperation.DiscoverModels };
}
