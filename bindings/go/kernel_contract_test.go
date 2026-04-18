package innovate_test

import (
	"reflect"
	"testing"

	innovate "github.com/edithatogo/innovate/bindings/go"
)

func TestKernelRequestEnvelopeShape(t *testing.T) {
	t.Helper()

	req := innovate.KernelRequest{
		SchemaVersion: innovate.KernelSchemaVersion(),
		Operation:     "discover_models",
		ModelKey:      nil,
		Payload:       map[string]any{},
		Metadata:      map[string]any{},
	}

	if got, want := req.Operation, "discover_models"; got != want {
		t.Fatalf("operation = %q, want %q", got, want)
	}
	if got, want := req.SchemaVersion, "1.0"; got != want {
		t.Fatalf("schema version = %q, want %q", got, want)
	}
}

func TestKernelResponseEnvelopeShape(t *testing.T) {
	t.Helper()

	resp := innovate.KernelResponse{
		SchemaVersion: innovate.KernelSchemaVersion(),
		Operation:     "fit_model",
		ModelKey:      "bass",
		Payload:       map[string]any{"status": "ok"},
		Metadata:      map[string]any{},
	}

	if got, want := resp.ModelKey, "bass"; got != want {
		t.Fatalf("model key = %q, want %q", got, want)
	}
}

func TestKernelErrorEnvelopeShape(t *testing.T) {
	t.Helper()

	err := innovate.KernelError{
		Code:      "invalid_request",
		Message:   "bad request",
		Operation: "fit_model",
		Retryable: false,
		Details:   map[string]any{},
	}

	if got, want := err.Code, "invalid_request"; got != want {
		t.Fatalf("code = %q, want %q", got, want)
	}
	if reflect.ValueOf(err.Details).Len() != 0 {
		t.Fatalf("expected empty error details")
	}
}
