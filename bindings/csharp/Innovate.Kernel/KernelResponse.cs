using System.Text.Json;
using System.Text.Json.Serialization;

namespace Innovate.Kernel;

/// <summary>Language-neutral kernel response envelope.</summary>
public sealed record KernelResponse
{
    [JsonPropertyName("schema_version")]
    public string SchemaVersion { get; init; } = "";

    [JsonPropertyName("operation")]
    public string Operation { get; init; } = "";

    [JsonPropertyName("model_key")]
    public string? ModelKey { get; init; }

    [JsonPropertyName("result")]
    public JsonElement? Result { get; init; }

    [JsonPropertyName("models")]
    public JsonElement? Models { get; init; }

    [JsonPropertyName("metadata")]
    public Dictionary<string, JsonElement> Metadata { get; init; } = [];

    [JsonPropertyName("error")]
    public KernelError? Error { get; init; }
}
