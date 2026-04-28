namespace Innovate.Kernel;

/// <summary>Stable operation names exposed by the shared kernel contract.</summary>
public static class KernelOperation
{
    public const string DiscoverModels = "discover_models";
    public const string FitModel = "fit_model";
    public const string PredictModel = "predict_model";
    public const string SimulateModel = "simulate_model";
    public const string SummarizeModel = "summarize_model";
    public const string DiagnoseModel = "diagnose_model";

    public static IReadOnlyList<string> All { get; } =
    [
        DiscoverModels,
        FitModel,
        PredictModel,
        SimulateModel,
        SummarizeModel,
        DiagnoseModel,
    ];
}
