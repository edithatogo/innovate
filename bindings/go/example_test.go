package innovate_test

import (
	"fmt"

	innovate "github.com/edithatogo/innovate/bindings/go"
)

func ExampleDiscoverModels() {
	time := []float64{0, 1, 2, 3, 4}
	observed := []float64{0.02, 0.06, 0.12, 0.25, 0.41}

	discovery, err := innovate.DiscoverModels()
	if err != nil {
		panic(err)
	}

	var bass innovate.KernelDiscoveryRecord
	for _, record := range discovery.Models {
		if record.Key == "bass" {
			bass = record
			break
		}
	}
	if bass.Key == "" {
		panic("bass model must be discoverable")
	}

	fit, err := innovate.FitModel(innovate.KernelRequest{
		SchemaVersion: innovate.KernelSchemaVersion(),
		Operation:     "fit_model",
		ModelKey:      &bass.Key,
		Payload: map[string]any{
			"inputs": map[string]any{
				"time":     time,
				"observed": observed,
			},
			"model_kwargs": map[string]any{},
		},
		Metadata: map[string]any{},
	})
	if err != nil {
		panic(err)
	}

	diagnostics, ok := innovate.ExtractDiagnostics(fit)
	if !ok {
		panic("fit response should expose diagnostics")
	}

	predict, err := innovate.PredictModel(innovate.KernelRequest{
		SchemaVersion: innovate.KernelSchemaVersion(),
		Operation:     "predict_model",
		ModelKey:      &bass.Key,
		Payload: map[string]any{
			"inputs": map[string]any{
				"time": time,
			},
			"state": fit.State,
		},
		Metadata: map[string]any{},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(discovery.SchemaVersion)
	fmt.Println(bass.Key)
	fmt.Println(diagnostics["support_level"])
	fmt.Println(predict.ModelKey)
	// Output:
	// 1.0
	// bass
	// supported
	// bass
}
