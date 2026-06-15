# Julia Binding Snippet

```julia
using Innovate

model = fit_model(
    "bass",
    Dict("time" => [1, 2, 3, 4], "adoption" => [3, 8, 15, 25]),
)

predictions = predict_model(
    model;
    horizon = 6,
    payload = Dict("schema_version" => "1.0"),
)
```

