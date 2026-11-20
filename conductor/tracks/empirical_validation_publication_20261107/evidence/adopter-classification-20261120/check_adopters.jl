using Innovate

report = Innovate.JSON.parsefile(only(ARGS))
cases = report["valid"]
for case in cases
    time = [reinterpret(Float64, parse(UInt64, bits; base=16)) for bits in case["time_bits"]]
    rates = [reinterpret(Float64, parse(UInt64, bits; base=16)) for bits in case["rate_bits"]]
    result = kernel_categorize_adopters(time, rates)
    result["time"] == time || error("$(case["name"]) changed time values")
    result["adoption_rate"] == rates || error("$(case["name"]) changed weights")
    result["category"] == case["categories"] || error("$(case["name"]) changed the labels")
end

for case in report["invalid"]
    observed = try
        kernel_categorize_adopters(Float64.(case["time"]), Float64.(case["rates"]))
        "Successful response"
    catch exception
        sprint(showerror, exception)
    end
    occursin(case["error"], observed) || error("$(case["name"]) returned unexpected result $observed")
end

case = report["logistic_fit"]
fit = kernel_fit_model("logistic",
    Dict("inputs" => Dict("time" => case["time"], "observed" => case["observed"]),
         "fit_options" => Dict("initial_parameters" => case["initial"])))["result"]
all(abs(fit["parameters"][key] - expected) <= 1e-8
    for (key, expected) in ("L" => 100.0, "k" => 1.0, "x0" => 10.0)) || error("Logistic fit differs")
classified = kernel_categorize_fitted_adopters("logistic",
    Dict("inputs" => Dict("time" => case["prediction_time"]), "state" => fit["state"]))
all(abs.(classified["adoption_rate"] .- case["analytic_rates"]) .<= 1e-8) || error("Logistic rates differ")
classified["category"] == case["categories"] || error("Logistic labels differ")
println("Julia ", VERSION, ": ", length(cases), " reference cases, ", length(report["invalid"]),
        " error cases and the fitted Logistic case passed")
