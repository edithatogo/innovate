import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { delimiter, join } from "node:path";
import { tmpdir } from "node:os";

import {
  kernelBridgeScript,
  kernelPackageRoot,
  kernelPythonCommand,
  kernelRepoRootOrNull,
  kernelRequest,
  type KernelRequest,
} from "./kernel.js";

export type KernelJSONValue =
  | string
  | number
  | boolean
  | null
  | KernelJSONValue[]
  | { [key: string]: KernelJSONValue };

export interface KernelDiscoveryRecord {
  key: string;
  family: string;
  import_path: string;
  stability: string;
  supports_covariates: boolean;
  supports_multivariate_output: boolean;
  supported_backends: string[];
  optional_dependencies: string[];
  supports_simulation: boolean;
  supports_summarize: boolean;
}

export interface KernelBridgeErrorDetails {
  [key: string]: KernelJSONValue;
}

export interface KernelBridgeErrorResponse {
  code: string;
  message: string;
  operation?: string | null;
  details?: KernelBridgeErrorDetails;
  retryable?: boolean;
}

export interface KernelFitResponse {
  model_key: string;
  model_name: string;
  family: string;
  parameters: Record<string, KernelJSONValue>;
  predictions: KernelJSONValue;
  diagnostics?: Record<string, KernelJSONValue>;
  state: Record<string, KernelJSONValue>;
}

export interface KernelSummaryResponse {
  model_key: string;
  model_name: string;
  family: string;
  parameter_names: string[];
  parameters: Record<string, KernelJSONValue>;
  constructor_kwargs: Record<string, KernelJSONValue>;
  state: Record<string, KernelJSONValue>;
  diagnostics?: Record<string, KernelJSONValue>;
}

export interface KernelDiagnoseResponse {
  diagnostics: Record<string, KernelJSONValue>;
  state: Record<string, KernelJSONValue>;
}

function splitCommand(command: string): string[] {
  const parts = command.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) {
    throw new Error("INNOVATE_PYTHON_COMMAND must not be empty");
  }
  return parts;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeKernelValue(value: unknown): KernelJSONValue {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }

  if (Array.isArray(value)) {
    return value.map((item) => normalizeKernelValue(item));
  }

  if (isRecord(value)) {
    if (
      Array.isArray(value.shape) &&
      value.shape.every((dimension) => typeof dimension === "number") &&
      Array.isArray(value.values)
    ) {
      const values = value.values.map((item) => normalizeKernelValue(item));
      return reshapeKernelArray(values, value.shape as number[]);
    }

    if (
      Array.isArray(value.columns) &&
      Array.isArray(value.rows) &&
      value.rows.every((row) => Array.isArray(row))
    ) {
      const columns = value.columns.map((column) => String(column));
      return value.rows.map((row) => {
        const entries = row as unknown[];
        return Object.fromEntries(
          columns.map((column, index) => [column, normalizeKernelValue(entries[index])]),
        );
      });
    }

    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, normalizeKernelValue(item)]),
    );
  }

  return value as never;
}

function reshapeKernelArray(values: KernelJSONValue[], shape: number[]): KernelJSONValue {
  if (shape.length === 0) {
    return values[0] ?? null;
  }

  const [dimension, ...rest] = shape;
  if (dimension === 0) {
    return [];
  }

  if (rest.length === 0) {
    return values.slice(0, dimension);
  }

  const chunkSize = rest.reduce((product, item) => product * item, 1);
  const chunks: KernelJSONValue[] = [];
  for (let index = 0; index < dimension; index += 1) {
    const start = index * chunkSize;
    const end = start + chunkSize;
    chunks.push(reshapeKernelArray(values.slice(start, end), rest));
  }
  return chunks;
}

function responseError(response: Record<string, unknown>): KernelBridgeError | null {
  if (!isRecord(response.error) || response.error == null) {
    return null;
  }

  return KernelBridgeError.fromResponse(response, response.operation);
}

function decodeBridgeResponse(response: Record<string, unknown>): KernelJSONValue {
  const error = responseError(response);
  if (error) {
    throw error;
  }

  if (Array.isArray(response.models) && !("operation" in response)) {
    return response.models.map((model) => normalizeKernelValue(model));
  }

  if ("result" in response) {
    return normalizeKernelValue(response.result);
  }

  return normalizeKernelValue(response);
}

function invokeBridge(request: KernelRequest): KernelJSONValue {
  const tempDir = mkdtempSync(join(tmpdir(), "innovate-typescript-kernel-"));
  const requestPath = join(tempDir, "request.json");
  const responsePath = join(tempDir, "response.json");
  const command = splitCommand(kernelPythonCommand());
  const repoRoot = kernelRepoRootOrNull();
  const env = {
    ...process.env,
    PYTHONPATH: [repoRoot ? join(repoRoot, "src") : null, process.env.PYTHONPATH]
      .filter((part) => Boolean(part))
      .join(delimiter),
  };

  try {
    writeFileSync(requestPath, `${JSON.stringify(request, null, 2)}\n`, "utf8");
    const result = spawnSync(command[0], [...command.slice(1), kernelBridgeScript(), requestPath, responsePath], {
      cwd: repoRoot ?? kernelPackageRoot(),
      env,
      encoding: "utf8",
    });

    if (result.error) {
      throw new KernelBridgeError("Kernel bridge process failed", {
        code: "internal_error",
        operation: request.operation,
        details: { stderr: result.stderr?.toString() ?? "", stdout: result.stdout?.toString() ?? "" },
        retryable: false,
      });
    }

    if (!existsSync(responsePath)) {
      throw new KernelBridgeError("Kernel bridge did not produce a response", {
        code: "internal_error",
        operation: request.operation,
        details: { stderr: result.stderr?.toString() ?? "", stdout: result.stdout?.toString() ?? "" },
        retryable: false,
      });
    }

    const response = JSON.parse(readFileSync(responsePath, "utf8")) as Record<string, unknown>;
    return decodeBridgeResponse(response);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

export class KernelBridgeError extends Error {
  readonly code: string;
  readonly operation: string;
  readonly details: KernelBridgeErrorDetails;
  readonly retryable: boolean;
  readonly response: Record<string, KernelJSONValue>;

  constructor(
    message: string,
    options: {
      code: string;
      operation: string;
      details?: KernelBridgeErrorDetails;
      retryable?: boolean;
      response?: Record<string, KernelJSONValue>;
    },
  ) {
    super(message);
    this.name = "KernelBridgeError";
    this.code = options.code;
    this.operation = options.operation;
    this.details = options.details ?? {};
    this.retryable = options.retryable ?? false;
    this.response = options.response ?? {};
    Object.setPrototypeOf(this, new.target.prototype);
  }

  static fromResponse(
    response: Record<string, unknown>,
    fallbackOperation?: unknown,
  ): KernelBridgeError {
    const error = isRecord(response.error) ? response.error : {};
    const operation =
      typeof error.operation === "string" && error.operation.length > 0
        ? error.operation
        : typeof fallbackOperation === "string" && fallbackOperation.length > 0
          ? fallbackOperation
          : "discover_models";
    const details = isRecord(error.details)
      ? Object.fromEntries(
          Object.entries(error.details).map(([key, value]) => [key, normalizeKernelValue(value)]),
        )
      : {};

    return new KernelBridgeError(String(error.message ?? "Kernel bridge error"), {
      code: String(error.code ?? "internal_error"),
      operation,
      details,
      retryable: Boolean(error.retryable ?? false),
      response: Object.fromEntries(
        Object.entries(response).map(([key, value]) => [key, normalizeKernelValue(value)]),
      ),
    });
  }
}

/** Discover the stable model registry through the kernel bridge. */
export function kernelDiscoverModels(): KernelDiscoveryRecord[] {
  return invokeBridge(kernelRequest("discover_models")) as unknown as KernelDiscoveryRecord[];
}

/** Fit a stable model through the kernel bridge. */
export function kernelFitModel(request: KernelRequest): KernelFitResponse {
  return invokeBridge(request) as unknown as KernelFitResponse;
}

/** Predict from a fitted stable model through the kernel bridge. */
export function kernelPredictModel(request: KernelRequest): KernelJSONValue {
  return invokeBridge(request);
}

/** Simulate a stable model through the kernel bridge. */
export function kernelSimulateModel(request: KernelRequest): KernelJSONValue {
  return invokeBridge(request);
}

/** Summarize a fitted stable model through the kernel bridge. */
export function kernelSummarizeModel(request: KernelRequest): KernelSummaryResponse {
  return invokeBridge(request) as unknown as KernelSummaryResponse;
}

/** Diagnose a fitted stable model through the kernel bridge. */
export function kernelDiagnoseModel(request: KernelRequest): KernelDiagnoseResponse {
  return invokeBridge(request) as unknown as KernelDiagnoseResponse;
}

/** Extract diagnostics from a kernel response payload when present. */
export function kernelExtractDiagnostics(
  result: unknown,
): Record<string, KernelJSONValue> | null {
  if (isRecord(result) && isRecord(result.diagnostics)) {
    return Object.fromEntries(
      Object.entries(result.diagnostics).map(([key, value]) => [key, normalizeKernelValue(value)]),
    );
  }

  return null;
}
