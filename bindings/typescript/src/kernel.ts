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
  if (process.env.INNOVATE_REPO_ROOT) {
    return resolve(process.env.INNOVATE_REPO_ROOT);
  }

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

export function kernelRepoRootOrNull(startDir = dirname(fileURLToPath(import.meta.url))): string | null {
  try {
    return kernelRepoRoot(startDir);
  } catch {
    return null;
  }
}

export function kernelPackageRoot(): string {
  const moduleDir = dirname(fileURLToPath(import.meta.url));
  return moduleDir.endsWith("dist") ? dirname(moduleDir) : dirname(moduleDir);
}

export function kernelBindingsRoot(): string {
  const repoRoot = kernelRepoRootOrNull();
  return repoRoot ? join(repoRoot, "bindings", "typescript") : kernelPackageRoot();
}

export function kernelBridgeScript(): string {
  const repoBridge = join(kernelBindingsRoot(), "inst", "python", "kernel_bridge.py");
  if (existsSync(repoBridge)) {
    return repoBridge;
  }

  const packagedBridge = join(kernelPackageRoot(), "inst", "python", "kernel_bridge.py");
  if (existsSync(packagedBridge)) {
    return packagedBridge;
  }

  throw new Error("Unable to locate the packaged Innovate kernel bridge");
}

export function kernelPythonCommand(): string {
  return process.env.INNOVATE_PYTHON_COMMAND || "uv run python";
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
