package innovate_test

import (
	"errors"
	"testing"

	innovate "github.com/edithatogo/innovate/bindings/go"
)

func stringPtr(value string) *string {
	return &value
}

func discoverBass(t *testing.T) innovate.KernelDiscoveryRecord {
	t.Helper()

	discovery, err := innovate.DiscoverModels()
	if err != nil {
		t.Fatalf("discover models: %v", err)
	}
	if discovery.SchemaVersion != innovate.KernelSchemaVersion() {
		t.Fatalf("discovery schema version = %q, want %q", discovery.SchemaVersion, innovate.KernelSchemaVersion())
	}
	for _, record := range discovery.Models {
		if record.Key == "bass" {
			return record
		}
	}

	t.Fatalf("expected bass to be discoverable")
	return innovate.KernelDiscoveryRecord{}
}

func kernelRequestFor(modelKey string, operation string, payload map[string]any) innovate.KernelRequest {
	return innovate.KernelRequest{
		SchemaVersion: innovate.KernelSchemaVersion(),
		Operation:     operation,
		ModelKey:      stringPtr(modelKey),
		Payload:       payload,
		Metadata:      map[string]any{},
	}
}

func TestKernelDiscoverModelsWrapper(t *testing.T) {
	t.Helper()

	bass := discoverBass(t)
	if bass.Family != "diffusion" {
		t.Fatalf("bass family = %q, want %q", bass.Family, "diffusion")
	}
	if len(bass.SupportedBackends) == 0 {
		t.Fatalf("expected bass to declare supported backends")
	}
}

func TestKernelStableWrappersRoundTrip(t *testing.T) {
	t.Helper()

	time := []float64{0, 1, 2, 3, 4}
	observed := []float64{0.02, 0.06, 0.12, 0.25, 0.41}
	bass := discoverBass(t)

	fit, err := innovate.FitModel(kernelRequestFor(
		bass.Key,
		"fit_model",
		map[string]any{
			"inputs": map[string]any{
				"time":     time,
				"observed": observed,
			},
			"model_kwargs": map[string]any{},
		},
	))
	if err != nil {
		t.Fatalf("fit model: %v", err)
	}
	if fit.ModelKey != bass.Key {
		t.Fatalf("fit model_key = %q, want %q", fit.ModelKey, bass.Key)
	}
	if fit.Family != bass.Family {
		t.Fatalf("fit family = %q, want %q", fit.Family, bass.Family)
	}
	if got := fit.Diagnostics["support_level"]; got != "supported" {
		t.Fatalf("fit support_level = %v, want supported", got)
	}
	if fit.State["model_key"] != bass.Key {
		t.Fatalf("fit state model_key = %v, want %q", fit.State["model_key"], bass.Key)
	}
	if fit.Predictions == nil {
		t.Fatalf("fit predictions must be populated")
	}
	if diagnostics, ok := innovate.ExtractDiagnostics(fit); !ok || diagnostics["support_level"] != "supported" {
		t.Fatalf("extract diagnostics mismatch: ok=%v diagnostics=%v", ok, diagnostics)
	}

	predict, err := innovate.PredictModel(kernelRequestFor(
		bass.Key,
		"predict_model",
		map[string]any{
			"inputs": map[string]any{
				"time": time,
			},
			"state": fit.State,
		},
	))
	if err != nil {
		t.Fatalf("predict model: %v", err)
	}
	if predict.ModelKey != bass.Key || predict.Family != bass.Family {
		t.Fatalf("predict metadata mismatch: %+v", predict)
	}
	if predict.Value == nil {
		t.Fatalf("predict value must be populated")
	}

	simulate, err := innovate.SimulateModel(kernelRequestFor(
		bass.Key,
		"simulate_model",
		map[string]any{
			"inputs": map[string]any{
				"time": time,
			},
			"state": fit.State,
		},
	))
	if err != nil {
		t.Fatalf("simulate model: %v", err)
	}
	if simulate.ModelKey != bass.Key || simulate.Family != bass.Family {
		t.Fatalf("simulate metadata mismatch: %+v", simulate)
	}
	if simulate.Value == nil {
		t.Fatalf("simulate value must be populated")
	}

	summary, err := innovate.SummarizeModel(kernelRequestFor(
		bass.Key,
		"summarize_model",
		map[string]any{
			"inputs": map[string]any{
				"time":     time,
				"observed": observed,
			},
			"state": fit.State,
		},
	))
	if err != nil {
		t.Fatalf("summarize model: %v", err)
	}
	if summary.ModelKey != bass.Key || summary.Family != bass.Family {
		t.Fatalf("summary metadata mismatch: %+v", summary)
	}
	if got := summary.Diagnostics["support_level"]; got != "supported" {
		t.Fatalf("summary support_level = %v, want supported", got)
	}
	if diagnostics, ok := innovate.ExtractDiagnostics(summary); !ok || diagnostics["support_level"] != "supported" {
		t.Fatalf("summary diagnostics mismatch: ok=%v diagnostics=%v", ok, diagnostics)
	}

	diagnose, err := innovate.DiagnoseModel(kernelRequestFor(
		bass.Key,
		"diagnose_model",
		map[string]any{
			"inputs": map[string]any{
				"time":     time,
				"observed": observed,
			},
			"state": fit.State,
		},
	))
	if err != nil {
		t.Fatalf("diagnose model: %v", err)
	}
	if diagnose.Diagnostics["support_level"] != "supported" {
		t.Fatalf("diagnose support_level = %v, want supported", diagnose.Diagnostics["support_level"])
	}
	if diagnose.State["model_key"] != bass.Key {
		t.Fatalf("diagnose state model_key = %v, want %q", diagnose.State["model_key"], bass.Key)
	}
	if diagnostics, ok := innovate.ExtractDiagnostics(diagnose); !ok || diagnostics["support_level"] != "supported" {
		t.Fatalf("diagnose diagnostics mismatch: ok=%v diagnostics=%v", ok, diagnostics)
	}
}

func TestKernelBridgeErrorMapping(t *testing.T) {
	t.Helper()

	bass := discoverBass(t)
	_, err := innovate.FitModel(kernelRequestFor(
		bass.Key,
		"fit_model",
		map[string]any{
			"inputs": map[string]any{
				"time": []float64{0, 1, 2, 3, 4},
			},
			"model_kwargs": map[string]any{},
		},
	))
	if err == nil {
		t.Fatalf("expected a kernel bridge error")
	}

	var bridgeErr *innovate.KernelBridgeError
	if !errors.As(err, &bridgeErr) {
		t.Fatalf("expected KernelBridgeError, got %T", err)
	}
	if bridgeErr.Code != "invalid_request" {
		t.Fatalf("bridge error code = %q, want %q", bridgeErr.Code, "invalid_request")
	}
	if bridgeErr.Operation != "fit_model" {
		t.Fatalf("bridge error operation = %q, want %q", bridgeErr.Operation, "fit_model")
	}
}
