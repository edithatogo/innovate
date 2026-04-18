package innovate

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

const kernelOperationDiscoverModels = "discover_models"
const kernelOperationFitModel = "fit_model"
const kernelOperationPredictModel = "predict_model"
const kernelOperationSimulateModel = "simulate_model"
const kernelOperationSummarizeModel = "summarize_model"
const kernelOperationDiagnoseModel = "diagnose_model"

// KernelJSONValue represents a JSON-compatible kernel payload value.
type KernelJSONValue = any

// KernelOperation names a stable kernel operation.
type KernelOperation string

// KernelBridgeError mirrors the stable error envelope returned by the kernel bridge.
type KernelBridgeError struct {
	Code      string
	Message   string
	Operation string
	Details   map[string]KernelJSONValue
	Retryable bool
	Response  map[string]KernelJSONValue
}

func (e *KernelBridgeError) Error() string {
	return fmt.Sprintf("kernel bridge error (%s) for operation %q: %s", e.Code, e.Operation, e.Message)
}

// KernelDiscoveryRecord describes a discoverable model family.
type KernelDiscoveryRecord struct {
	Key                        string   `json:"key"`
	Family                     string   `json:"family"`
	ImportPath                 string   `json:"import_path"`
	Stability                  string   `json:"stability"`
	SupportsCovariates         bool     `json:"supports_covariates"`
	SupportsMultivariateOutput bool     `json:"supports_multivariate_output"`
	SupportedBackends          []string `json:"supported_backends"`
	OptionalDependencies       []string `json:"optional_dependencies,omitempty"`
	SupportsSimulation         bool     `json:"supports_simulation"`
	SupportsSummarize          bool     `json:"supports_summarize"`
}

// KernelDiscoveryResponse contains the discoverable model registry and bridge metadata.
type KernelDiscoveryResponse struct {
	SchemaVersion string                     `json:"schema_version"`
	Models        []KernelDiscoveryRecord    `json:"models"`
	Metadata      map[string]KernelJSONValue `json:"metadata,omitempty"`
}

// KernelPredictionResponse contains a normalized prediction or simulation result.
type KernelPredictionResponse struct {
	SchemaVersion string                     `json:"schema_version"`
	Operation     string                     `json:"operation"`
	ModelKey      string                     `json:"model_key"`
	ModelName     string                     `json:"model_name"`
	Family        string                     `json:"family"`
	Value         KernelJSONValue            `json:"value"`
	Metadata      map[string]KernelJSONValue `json:"metadata,omitempty"`
}

// KernelFitResponse contains the fitted model payload and diagnostics.
type KernelFitResponse struct {
	SchemaVersion string                     `json:"schema_version"`
	Operation     string                     `json:"operation"`
	ModelKey      string                     `json:"model_key"`
	ModelName     string                     `json:"model_name"`
	Family        string                     `json:"family"`
	Parameters    map[string]KernelJSONValue `json:"parameters"`
	Predictions   KernelJSONValue            `json:"predictions"`
	Diagnostics   map[string]KernelJSONValue `json:"diagnostics,omitempty"`
	State         map[string]KernelJSONValue `json:"state"`
	Metadata      map[string]KernelJSONValue `json:"metadata,omitempty"`
}

// KernelSummaryResponse contains a summarized fitted model payload.
type KernelSummaryResponse struct {
	SchemaVersion     string                     `json:"schema_version"`
	Operation         string                     `json:"operation"`
	ModelKey          string                     `json:"model_key"`
	ModelName         string                     `json:"model_name"`
	Family            string                     `json:"family"`
	ParameterNames    []string                   `json:"parameter_names"`
	Parameters        map[string]KernelJSONValue `json:"parameters"`
	ConstructorKwargs map[string]KernelJSONValue `json:"constructor_kwargs"`
	State             map[string]KernelJSONValue `json:"state"`
	Diagnostics       map[string]KernelJSONValue `json:"diagnostics,omitempty"`
	Metadata          map[string]KernelJSONValue `json:"metadata,omitempty"`
}

// KernelDiagnoseResponse contains a structured diagnostics contract for a fitted model.
type KernelDiagnoseResponse struct {
	SchemaVersion string                     `json:"schema_version"`
	Operation     string                     `json:"operation"`
	Diagnostics   map[string]KernelJSONValue `json:"diagnostics"`
	State         map[string]KernelJSONValue `json:"state"`
	Metadata      map[string]KernelJSONValue `json:"metadata,omitempty"`
}

func repoRoot() string {
	return filepath.Clean(filepath.Join(ModuleRoot(), "..", ".."))
}

func kernelCommandParts() ([]string, error) {
	command := strings.TrimSpace(os.Getenv("INNOVATE_PYTHON_COMMAND"))
	if command != "" {
		parts := strings.Fields(command)
		if len(parts) == 0 {
			return nil, errors.New("INNOVATE_PYTHON_COMMAND must not be empty")
		}
		return parts, nil
	}

	if _, err := exec.LookPath("uv"); err == nil {
		return []string{"uv", "run", "python"}, nil
	}
	if _, err := exec.LookPath("python3"); err == nil {
		return []string{"python3"}, nil
	}
	return nil, errors.New("unable to locate a Python launcher for the kernel bridge")
}

func kernelInvokeBridge(request KernelRequest) (map[string]KernelJSONValue, error) {
	tempDir, err := os.MkdirTemp("", "innovate-go-kernel-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tempDir)

	requestPath := filepath.Join(tempDir, "request.json")
	responsePath := filepath.Join(tempDir, "response.json")

	requestBytes, err := json.MarshalIndent(request, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(requestPath, append(requestBytes, '\n'), 0o600); err != nil {
		return nil, err
	}

	commandParts, err := kernelCommandParts()
	if err != nil {
		return nil, err
	}

	args := append([]string{}, commandParts[1:]...)
	args = append(args, BridgeScriptPath(), requestPath, responsePath)
	cmd := exec.Command(commandParts[0], args...)
	cmd.Dir = repoRoot()
	cmd.Env = append(os.Environ(), "PYTHONPATH="+strings.Join([]string{filepath.Join(repoRoot(), "src"), os.Getenv("PYTHONPATH")}, string(os.PathListSeparator)))

	output, runErr := cmd.CombinedOutput()
	if runErr != nil {
		return nil, &KernelBridgeError{
			Code:      "internal_error",
			Message:   "Kernel bridge process failed",
			Operation: request.Operation,
			Details: map[string]KernelJSONValue{
				"stdout": strings.TrimSpace(string(output)),
			},
			Retryable: false,
			Response:  map[string]KernelJSONValue{},
		}
	}

	responseBytes, err := os.ReadFile(responsePath)
	if err != nil {
		return nil, &KernelBridgeError{
			Code:      "internal_error",
			Message:   "Kernel bridge did not produce a response",
			Operation: request.Operation,
			Details: map[string]KernelJSONValue{
				"stdout": strings.TrimSpace(string(output)),
			},
			Retryable: false,
			Response:  map[string]KernelJSONValue{},
		}
	}

	var response map[string]KernelJSONValue
	if err := json.Unmarshal(responseBytes, &response); err != nil {
		return nil, err
	}

	if bridgeErr := kernelBridgeErrorFromResponse(response, request.Operation); bridgeErr != nil {
		return nil, bridgeErr
	}

	return response, nil
}

func kernelBridgeErrorFromResponse(response map[string]KernelJSONValue, fallbackOperation string) *KernelBridgeError {
	rawError, ok := response["error"].(map[string]any)
	if !ok || rawError == nil {
		return nil
	}

	operation := fallbackOperation
	if value, ok := rawError["operation"].(string); ok && value != "" {
		operation = value
	}

	details := map[string]KernelJSONValue{}
	if rawDetails, ok := rawError["details"].(map[string]any); ok && rawDetails != nil {
		details = normalizeMap(rawDetails)
	}

	retryable := false
	if value, ok := rawError["retryable"].(bool); ok {
		retryable = value
	}

	code, _ := rawError["code"].(string)
	message, _ := rawError["message"].(string)

	return &KernelBridgeError{
		Code:      code,
		Message:   message,
		Operation: operation,
		Details:   details,
		Retryable: retryable,
		Response:  normalizeMap(response),
	}
}

func normalizeKernelValue(value any) KernelJSONValue {
	switch typed := value.(type) {
	case nil, string, bool:
		return typed
	case float32:
		return float64(typed)
	case float64, int, int32, int64, uint32, uint64:
		return typed
	case []any:
		values := make([]KernelJSONValue, len(typed))
		for index, item := range typed {
			values[index] = normalizeKernelValue(item)
		}
		return values
	case map[string]any:
		if shape, ok := kernelShapeFromValue(typed["shape"]); ok {
			if values, ok := typed["values"].([]any); ok {
				normalized := make([]KernelJSONValue, len(values))
				for index, item := range values {
					normalized[index] = normalizeKernelValue(item)
				}
				return reshapeKernelArray(normalized, shape)
			}
		}

		if columns, ok := typed["columns"].([]any); ok {
			if rows, ok := typed["rows"].([]any); ok {
				columnNames := make([]string, 0, len(columns))
				for _, column := range columns {
					columnNames = append(columnNames, fmt.Sprint(column))
				}
				table := make([]map[string]KernelJSONValue, 0, len(rows))
				for _, rowValue := range rows {
					row, _ := rowValue.([]any)
					entry := make(map[string]KernelJSONValue, len(columnNames))
					for index, column := range columnNames {
						if index < len(row) {
							entry[column] = normalizeKernelValue(row[index])
						} else {
							entry[column] = nil
						}
					}
					table = append(table, entry)
				}
				return table
			}
		}

		return normalizeMap(typed)
	default:
		return typed
	}
}

func normalizeMap(values map[string]any) map[string]KernelJSONValue {
	normalized := make(map[string]KernelJSONValue, len(values))
	for key, value := range values {
		normalized[key] = normalizeKernelValue(value)
	}
	return normalized
}

func kernelShapeFromValue(value any) ([]int, bool) {
	rawShape, ok := value.([]any)
	if !ok {
		return nil, false
	}

	shape := make([]int, 0, len(rawShape))
	for _, item := range rawShape {
		switch dimension := item.(type) {
		case float64:
			shape = append(shape, int(dimension))
		case float32:
			shape = append(shape, int(dimension))
		case int:
			shape = append(shape, dimension)
		case int64:
			shape = append(shape, int(dimension))
		case uint64:
			shape = append(shape, int(dimension))
		case string:
			value, err := strconv.Atoi(dimension)
			if err != nil {
				return nil, false
			}
			shape = append(shape, value)
		default:
			return nil, false
		}
	}
	return shape, true
}

func reshapeKernelArray(values []KernelJSONValue, shape []int) KernelJSONValue {
	if len(shape) == 0 {
		if len(values) == 0 {
			return nil
		}
		return values[0]
	}

	dimension := shape[0]
	if dimension == 0 {
		return []KernelJSONValue{}
	}

	if len(shape) == 1 {
		end := dimension
		if end > len(values) {
			end = len(values)
		}
		return values[:end]
	}

	chunkSize := 1
	for _, item := range shape[1:] {
		chunkSize *= item
	}

	chunks := make([]KernelJSONValue, 0, dimension)
	for index := 0; index < dimension; index++ {
		start := index * chunkSize
		end := start + chunkSize
		if start >= len(values) {
			chunks = append(chunks, reshapeKernelArray(nil, shape[1:]))
			continue
		}
		if end > len(values) {
			end = len(values)
		}
		chunks = append(chunks, reshapeKernelArray(values[start:end], shape[1:]))
	}

	return chunks
}

func discoveryResponseFromMap(response map[string]KernelJSONValue) (KernelDiscoveryResponse, error) {
	rawModels, ok := response["models"].([]any)
	if !ok {
		return KernelDiscoveryResponse{}, errors.New("kernel discovery response must include models")
	}

	models := make([]KernelDiscoveryRecord, 0, len(rawModels))
	for _, rawModel := range rawModels {
		record, ok := rawModel.(map[string]any)
		if !ok {
			return KernelDiscoveryResponse{}, errors.New("kernel discovery records must be objects")
		}

		model := KernelDiscoveryRecord{
			Key:                        fmt.Sprint(record["key"]),
			Family:                     fmt.Sprint(record["family"]),
			ImportPath:                 fmt.Sprint(record["import_path"]),
			Stability:                  fmt.Sprint(record["stability"]),
			SupportsCovariates:         asBool(record["supports_covariates"]),
			SupportsMultivariateOutput: asBool(record["supports_multivariate_output"]),
			SupportedBackends:          stringSlice(record["supported_backends"]),
			OptionalDependencies:       stringSlice(record["optional_dependencies"]),
			SupportsSimulation:         asBool(record["supports_simulation"]),
			SupportsSummarize:          asBool(record["supports_summarize"]),
		}
		models = append(models, model)
	}

	return KernelDiscoveryResponse{
		SchemaVersion: fmt.Sprint(response["schema_version"]),
		Models:        models,
		Metadata:      mapFromKernelValue(response["metadata"]),
	}, nil
}

func mapFromKernelValue(value KernelJSONValue) map[string]KernelJSONValue {
	if value == nil {
		return map[string]KernelJSONValue{}
	}
	typed, ok := value.(map[string]KernelJSONValue)
	if ok {
		return typed
	}
	if raw, ok := value.(map[string]any); ok {
		return normalizeMap(raw)
	}
	return map[string]KernelJSONValue{}
}

func stringSlice(value KernelJSONValue) []string {
	raw, ok := value.([]any)
	if !ok {
		return nil
	}
	values := make([]string, 0, len(raw))
	for _, item := range raw {
		values = append(values, fmt.Sprint(item))
	}
	return values
}

func asBool(value any) bool {
	typed, ok := value.(bool)
	return ok && typed
}

// ExtractDiagnostics returns the diagnostics payload from a fitted, summarized, or diagnosed result.
func ExtractDiagnostics(result any) (map[string]KernelJSONValue, bool) {
	switch typed := result.(type) {
	case KernelFitResponse:
		return typed.Diagnostics, typed.Diagnostics != nil
	case *KernelFitResponse:
		if typed == nil {
			return nil, false
		}
		return typed.Diagnostics, typed.Diagnostics != nil
	case KernelSummaryResponse:
		return typed.Diagnostics, typed.Diagnostics != nil
	case *KernelSummaryResponse:
		if typed == nil {
			return nil, false
		}
		return typed.Diagnostics, typed.Diagnostics != nil
	case KernelDiagnoseResponse:
		return typed.Diagnostics, typed.Diagnostics != nil
	case *KernelDiagnoseResponse:
		if typed == nil {
			return nil, false
		}
		return typed.Diagnostics, typed.Diagnostics != nil
	case map[string]KernelJSONValue:
		if diagnostics, ok := typed["diagnostics"]; ok {
			return mapFromKernelValue(diagnostics), true
		}
	}

	return nil, false
}

func kernelResultEnvelope(response map[string]KernelJSONValue) (map[string]KernelJSONValue, error) {
	rawResult, ok := response["result"]
	if !ok || rawResult == nil {
		return map[string]KernelJSONValue{}, errors.New("kernel response did not include a result")
	}

	result, ok := rawResult.(map[string]any)
	if !ok {
		return map[string]KernelJSONValue{"value": normalizeKernelValue(rawResult)}, nil
	}

	return normalizeMap(result), nil
}

// DiscoverModels returns the registry of discoverable kernel models.
func DiscoverModels() (KernelDiscoveryResponse, error) {
	response, err := kernelInvokeBridge(KernelRequest{
		SchemaVersion: KernelSchemaVersion(),
		Operation:     kernelOperationDiscoverModels,
		ModelKey:      nil,
		Payload:       map[string]any{},
		Metadata:      map[string]any{},
	})
	if err != nil {
		return KernelDiscoveryResponse{}, err
	}

	return discoveryResponseFromMap(response)
}

// FitModel fits a stable model using the shared Python kernel.
func FitModel(request KernelRequest) (KernelFitResponse, error) {
	response, err := kernelInvokeBridge(request)
	if err != nil {
		return KernelFitResponse{}, err
	}

	result, err := kernelResultEnvelope(response)
	if err != nil {
		return KernelFitResponse{}, err
	}

	return KernelFitResponse{
		SchemaVersion: fmt.Sprint(response["schema_version"]),
		Operation:     fmt.Sprint(response["operation"]),
		ModelKey:      fmt.Sprint(result["model_key"]),
		ModelName:     fmt.Sprint(result["model_name"]),
		Family:        fmt.Sprint(result["family"]),
		Parameters:    mapFromKernelValue(result["parameters"]),
		Predictions:   normalizeKernelValue(result["predictions"]),
		Diagnostics:   mapFromKernelValue(result["diagnostics"]),
		State:         mapFromKernelValue(result["state"]),
		Metadata:      mapFromKernelValue(response["metadata"]),
	}, nil
}

// PredictModel predicts from a fitted stable model.
func PredictModel(request KernelRequest) (KernelPredictionResponse, error) {
	response, err := kernelInvokeBridge(request)
	if err != nil {
		return KernelPredictionResponse{}, err
	}

	metadata := mapFromKernelValue(response["metadata"])
	return KernelPredictionResponse{
		SchemaVersion: fmt.Sprint(response["schema_version"]),
		Operation:     fmt.Sprint(response["operation"]),
		ModelKey:      fmt.Sprint(metadata["model_key"]),
		ModelName:     fmt.Sprint(metadata["model_name"]),
		Family:        fmt.Sprint(metadata["family"]),
		Value:         normalizeKernelValue(response["result"]),
		Metadata:      metadata,
	}, nil
}

// SimulateModel simulates a fitted stable model.
func SimulateModel(request KernelRequest) (KernelPredictionResponse, error) {
	response, err := kernelInvokeBridge(request)
	if err != nil {
		return KernelPredictionResponse{}, err
	}

	metadata := mapFromKernelValue(response["metadata"])
	return KernelPredictionResponse{
		SchemaVersion: fmt.Sprint(response["schema_version"]),
		Operation:     fmt.Sprint(response["operation"]),
		ModelKey:      fmt.Sprint(metadata["model_key"]),
		ModelName:     fmt.Sprint(metadata["model_name"]),
		Family:        fmt.Sprint(metadata["family"]),
		Value:         normalizeKernelValue(response["result"]),
		Metadata:      metadata,
	}, nil
}

// SummarizeModel summarizes a fitted stable model.
func SummarizeModel(request KernelRequest) (KernelSummaryResponse, error) {
	response, err := kernelInvokeBridge(request)
	if err != nil {
		return KernelSummaryResponse{}, err
	}

	result, err := kernelResultEnvelope(response)
	if err != nil {
		return KernelSummaryResponse{}, err
	}

	return KernelSummaryResponse{
		SchemaVersion:     fmt.Sprint(response["schema_version"]),
		Operation:         fmt.Sprint(response["operation"]),
		ModelKey:          fmt.Sprint(result["model_key"]),
		ModelName:         fmt.Sprint(result["model_name"]),
		Family:            fmt.Sprint(result["family"]),
		ParameterNames:    stringSlice(result["parameter_names"]),
		Parameters:        mapFromKernelValue(result["parameters"]),
		ConstructorKwargs: mapFromKernelValue(result["constructor_kwargs"]),
		State:             mapFromKernelValue(result["state"]),
		Diagnostics:       mapFromKernelValue(result["diagnostics"]),
		Metadata:          mapFromKernelValue(response["metadata"]),
	}, nil
}

// DiagnoseModel returns a structured diagnostics contract for a fitted stable model.
func DiagnoseModel(request KernelRequest) (KernelDiagnoseResponse, error) {
	response, err := kernelInvokeBridge(request)
	if err != nil {
		return KernelDiagnoseResponse{}, err
	}

	result, err := kernelResultEnvelope(response)
	if err != nil {
		return KernelDiagnoseResponse{}, err
	}

	return KernelDiagnoseResponse{
		SchemaVersion: fmt.Sprint(response["schema_version"]),
		Operation:     fmt.Sprint(response["operation"]),
		Diagnostics:   mapFromKernelValue(result["diagnostics"]),
		State:         mapFromKernelValue(result["state"]),
		Metadata:      mapFromKernelValue(response["metadata"]),
	}, nil
}
