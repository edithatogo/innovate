import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const KERNEL_SCHEMA_VERSION = "1.0" as const;

export type KernelOperation =
  | "discover_models"
  | "fit_model"
  | "predict_model"
  | "simulate_model"
  | "summarize_model"
  | "diagnose_model";

export interface KernelRequestPayload {
  [key: string]: unknown;
}

export interface KernelRequestMetadata {
  [key: string]: unknown;
}

export interface KernelRequest {
  schema_version: string;
  operation: KernelOperation;
  model_key: string | null;
  payload: KernelRequestPayload;
  metadata: KernelRequestMetadata;
}

function hasKernelRoot(candidate: string): boolean {
  return (
    existsSync(join(candidate, "src", "innovate", "kernel.py")) &&
    existsSync(join(candidate, "conductor", "tracks.md"))
  );
}

export function kernelRepoRoot(startDir = dirname(fileURLToPath(import.meta.url))): string {
  let current = resolve(startDir);
  while (true) {
    if (hasKernelRoot(current)) {
      return current;
    }

    const parent = dirname(current);
    if (parent === current) {
      throw new Error("Unable to locate the innovate repository root");
    }

    current = parent;
  }
}

export function kernelBindingsRoot(): string {
  return join(kernelRepoRoot(), "bindings", "typescript");
}

export function kernelBridgeScript(): string {
  return join(kernelBindingsRoot(), "inst", "python", "kernel_bridge.py");
}

export function kernelPythonCommand(): string {
  return process.env.INNOVATE_PYTHON_COMMAND || "uv";
}

export function kernelSchemaVersion(): string {
  return KERNEL_SCHEMA_VERSION;
}

export function kernelRequest(
  operation: KernelOperation,
  modelKey: string | null = null,
  payload: KernelRequestPayload = {},
  metadata: KernelRequestMetadata = {},
  schemaVersion: string = KERNEL_SCHEMA_VERSION,
): KernelRequest {
  if (!operation) {
    throw new Error("kernelRequest() requires a non-empty operation");
  }

  if (operation !== "discover_models" && !modelKey) {
    throw new Error(`Kernel operation '${operation}' requires a model_key`);
  }

  return {
    schema_version: schemaVersion,
    operation,
    model_key: modelKey,
    payload,
    metadata,
  };
}
