package innovate

import (
	"path/filepath"
	"runtime"
)

const kernelSchemaVersionValue = "1.0"

// ModuleRoot returns the Go bindings module root on disk.
func ModuleRoot() string {
	return moduleRoot()
}

// BridgeScriptPath returns the Python bridge entrypoint used by the Go layer.
func BridgeScriptPath() string {
	return filepath.Join(moduleRoot(), "inst", "python", "kernel_bridge.py")
}

// KernelSchemaVersion returns the shared kernel contract schema version.
func KernelSchemaVersion() string {
	return kernelSchemaVersionValue
}

// KernelOperations returns the stable kernel operations exposed by the Go bindings.
func KernelOperations() []KernelOperation {
	return []KernelOperation{
		KernelOperation(kernelOperationDiscoverModels),
		KernelOperation(kernelOperationFitModel),
		KernelOperation(kernelOperationPredictModel),
		KernelOperation(kernelOperationSimulateModel),
		KernelOperation(kernelOperationSummarizeModel),
		KernelOperation(kernelOperationDiagnoseModel),
	}
}

func moduleRoot() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return filepath.FromSlash("bindings/go")
	}
	return filepath.Dir(file)
}

// KernelRequest mirrors the language-neutral kernel request envelope.
type KernelRequest struct {
	SchemaVersion string         `json:"schema_version"`
	Operation     string         `json:"operation"`
	ModelKey      *string        `json:"model_key"`
	Payload       map[string]any `json:"payload"`
	Metadata      map[string]any `json:"metadata"`
}

// KernelResponse mirrors the language-neutral kernel response envelope.
type KernelResponse struct {
	SchemaVersion string         `json:"schema_version"`
	Operation     string         `json:"operation"`
	ModelKey      string         `json:"model_key"`
	Payload       map[string]any `json:"payload"`
	Metadata      map[string]any `json:"metadata"`
	Error         *KernelError   `json:"error,omitempty"`
}

// KernelError mirrors the stable error envelope returned by the kernel bridge.
type KernelError struct {
	Code      string         `json:"code"`
	Message   string         `json:"message"`
	Operation string         `json:"operation,omitempty"`
	Details   map[string]any `json:"details"`
	Retryable bool           `json:"retryable"`
}
