namespace Innovate.Kernel;

/// <summary>Raised when the shared kernel bridge returns an error envelope.</summary>
public sealed class KernelBridgeException : Exception
{
    public KernelBridgeException(KernelError error, KernelResponse response)
        : base($"Kernel bridge error ({error.Code}) for operation '{error.Operation}': {error.Message}")
    {
        Error = error;
        Response = response;
    }

    public KernelError Error { get; }

    public KernelResponse Response { get; }
}
