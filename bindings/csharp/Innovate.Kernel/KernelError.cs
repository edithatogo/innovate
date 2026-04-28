using System.Text.Json;
using System.Text.Json.Serialization;

namespace Innovate.Kernel;

/// <summary>Stable kernel error payload.</summary>
public sealed record KernelError
{
    [JsonPropertyName("code")]
    public string Code { get; init; } = "";

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";

    [JsonPropertyName("operation")]
    public string Operation { get; init; } = "";

    [JsonPropertyName("details")]
    public Dictionary<string, JsonElement> Details { get; init; } = [];

    [JsonPropertyName("retryable")]
    public bool Retryable { get; init; }
}
