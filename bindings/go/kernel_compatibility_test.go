package innovate_test

import (
	"reflect"
	"testing"

	innovate "github.com/edithatogo/innovate/bindings/go"
)

func TestKernelCompatibilityDriftGuard(t *testing.T) {
	t.Helper()

	discovery, err := innovate.DiscoverModels()
	if err != nil {
		t.Fatalf("discover models: %v", err)
	}

	if got, want := discovery.SchemaVersion, innovate.KernelSchemaVersion(); got != want {
		t.Fatalf("schema version = %q, want %q", got, want)
	}

	wantOperations := []innovate.KernelOperation{
		"discover_models",
		"fit_model",
		"predict_model",
		"simulate_model",
		"summarize_model",
		"diagnose_model",
	}
	if got := innovate.KernelOperations(); !reflect.DeepEqual(got, wantOperations) {
		t.Fatalf("kernel operations = %#v, want %#v", got, wantOperations)
	}

	if bass := discoverBass(t); bass.Key == "" {
		t.Fatalf("expected bass to remain discoverable")
	}
}
