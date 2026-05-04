using System.Diagnostics;
using System.Text.Json;

namespace Innovate.Kernel;

/// <summary>Thin process bridge into the shared Python kernel contract.</summary>
public static class KernelBinding
{
    public const string SchemaVersion = "1.0";

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
    };

    public static string RepositoryRoot() =>
        Environment.GetEnvironmentVariable("INNOVATE_REPO_ROOT")
        ?? FindRepositoryRoot(Environment.CurrentDirectory)
        ?? FindRepositoryRoot(AppContext.BaseDirectory)
        ?? throw new InvalidOperationException("Unable to locate the innovate repository root.");

    public static string BindingRoot() =>
        TryFindRepositoryRoot() is { } repoRoot
            ? Path.Combine(repoRoot, "bindings", "csharp")
            : AppContext.BaseDirectory;

    public static string BridgeScriptPath()
    {
        var repoBridge = TryFindRepositoryRoot() is { } repoRoot
            ? Path.Combine(repoRoot, "bindings", "csharp", "inst", "python", "kernel_bridge.py")
            : null;
        if (repoBridge is not null && File.Exists(repoBridge))
        {
            return repoBridge;
        }

        var packagedBridge = Path.Combine(AppContext.BaseDirectory, "innovate", "kernel_bridge.py");
        if (File.Exists(packagedBridge))
        {
            return packagedBridge;
        }

        throw new FileNotFoundException("Unable to locate the Innovate kernel bridge.", packagedBridge);
    }

    public static string PythonCommand() =>
        string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("INNOVATE_PYTHON_COMMAND"))
            ? "uv run python"
            : Environment.GetEnvironmentVariable("INNOVATE_PYTHON_COMMAND")!;

    public static async Task<KernelResponse> CallAsync(KernelRequest request, CancellationToken cancellationToken = default)
    {
        var tempDirectory = Directory.CreateTempSubdirectory("innovate-csharp-kernel-");
        try
        {
            var requestPath = Path.Combine(tempDirectory.FullName, "request.json");
            var responsePath = Path.Combine(tempDirectory.FullName, "response.json");

            await File.WriteAllTextAsync(
                requestPath,
                JsonSerializer.Serialize(request, JsonOptions) + Environment.NewLine,
                cancellationToken
            );

            await RunBridgeAsync(request, requestPath, responsePath, cancellationToken);

            var responseJson = await File.ReadAllTextAsync(responsePath, cancellationToken);
            var response = JsonSerializer.Deserialize<KernelResponse>(responseJson, JsonOptions)
                ?? throw new InvalidOperationException("Kernel bridge returned an empty response.");

            if (response.Error is not null)
            {
                throw new KernelBridgeException(response.Error, response);
            }

            return response;
        }
        finally
        {
            tempDirectory.Delete(recursive: true);
        }
    }

    public static Task<KernelResponse> DiscoverModelsAsync(CancellationToken cancellationToken = default) =>
        CallAsync(KernelRequest.DiscoverModels(), cancellationToken);

    private static async Task RunBridgeAsync(
        KernelRequest request,
        string requestPath,
        string responsePath,
        CancellationToken cancellationToken
    )
    {
        var command = SplitCommand(PythonCommand());
        if (command.Count == 0)
        {
            throw new InvalidOperationException("INNOVATE_PYTHON_COMMAND must not be empty.");
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = command[0],
            WorkingDirectory = TryFindRepositoryRoot() ?? AppContext.BaseDirectory,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
        };

        foreach (var argument in command.Skip(1))
        {
            startInfo.ArgumentList.Add(argument);
        }

        startInfo.ArgumentList.Add(BridgeScriptPath());
        startInfo.ArgumentList.Add(requestPath);
        startInfo.ArgumentList.Add(responsePath);

        if (TryFindRepositoryRoot() is { } repoRoot)
        {
            var srcPath = Path.Combine(repoRoot, "src");
            var currentPythonPath = Environment.GetEnvironmentVariable("PYTHONPATH");
            startInfo.Environment["PYTHONPATH"] = string.IsNullOrEmpty(currentPythonPath)
                ? srcPath
                : srcPath + Path.PathSeparator + currentPythonPath;
        }

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Unable to start the kernel bridge process.");

        var stdoutTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
        var stderrTask = process.StandardError.ReadToEndAsync(cancellationToken);

        await process.WaitForExitAsync(cancellationToken);
        if (process.ExitCode != 0)
        {
            var stdout = await stdoutTask;
            var stderr = await stderrTask;
            throw new InvalidOperationException(
                $"Kernel bridge failed for operation '{request.Operation}' with exit code {process.ExitCode}: "
                + $"{stdout}{stderr}"
            );
        }
    }

    private static string? FindRepositoryRoot(string startDirectory)
    {
        var directory = new DirectoryInfo(startDirectory);
        while (directory is not null)
        {
            if (
                File.Exists(Path.Combine(directory.FullName, "src", "innovate", "kernel.py"))
                && Directory.Exists(Path.Combine(directory.FullName, "bindings", "csharp"))
            )
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    private static string? TryFindRepositoryRoot() =>
        Environment.GetEnvironmentVariable("INNOVATE_REPO_ROOT")
        ?? FindRepositoryRoot(Environment.CurrentDirectory)
        ?? FindRepositoryRoot(AppContext.BaseDirectory);

    private static IReadOnlyList<string> SplitCommand(string command) =>
        command.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
}
