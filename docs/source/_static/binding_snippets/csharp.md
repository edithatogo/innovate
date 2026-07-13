# C# Binding Snippet

```csharp
using Innovate.Kernel;

var model = await KernelBinding.FitModelAsync(
    "bass",
    new Dictionary<string, double[]>
    {
        ["time"] = [1, 2, 3, 4],
        ["adoption"] = [3, 8, 15, 25],
    });

var predictions = await KernelBinding.PredictModelAsync(
    model,
    new PredictOptions { Horizon = 6, SchemaVersion = "1.0" });
```
